# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native LeRobot engine, isolated from the Studio host environment."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Executes the framework's installed native script with shell=False.  # nosec B404
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module, metadata, resources  # nosemgrep  # Python 3.10+ only.
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from packaging.version import Version
from physicalai.engine import (
    Enablement,
    Run,
    TrainHooks,
    TrainingEngine,
    WorkerEnv,
    interpreter_for,
    locked_version,
    resolve_accelerator,
    run_worker,
    scripts_dir,
    supported_versions,
)

from .environments import (
    LEROBOT_EXPORT_UPSTREAM_REF,
    export_family_for_policy,
    export_worker_env,
    openvino_enablement_resource,
    training_worker_env,
)
from .policy_support import (
    SUPPORTED_POLICY_TYPES,
    VERIFIED_EXPORT_POLICIES,
    require_exportable_policy,
    require_supported_policy,
)

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_POLICY_NAME_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_SETTING_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*")
_MAX_SETTING_DEPTH = 10
_HUB_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
# Tells the worker script to run lerobot-train with inherited streams instead of the host protocol.
_NATIVE_CLI_VARIABLE = "PHYSICALAI_LEROBOT_NATIVE_CLI"
_ENABLEMENT_OV_DIR_ENV = "PHYSICALAI_LEROBOT_ENABLEMENT_OV_DIR"
# LeRobot trains `reward_model` instead of `policy` when one is set; the worker also checks this.
_REWARD_MODEL_UNSUPPORTED = "LeRobot reward-model training is not supported; set only `policy` options."
_EXPORT_BACKENDS = frozenset({"native", "openvino", "executorch_openvino"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeRobotExportOptions:
    """Options controlling isolated ``lerobot-export`` compilation and packaging.

    Attributes:
        task: Task prompt for task-conditioned policies.
        dataset: Optional Hugging Face Hub dataset ``repo_id`` or local dataset directory.
        dtype: Optional policy dtype override during export (``float32``, ``bfloat16``, ``float16``).
        device: Target OpenVINO device used for export-time self-check (e.g. ``"CPU"``, ``"GPU"``).
        quantize: Optional weight compression mode (e.g. ``"fp16"``, ``"int8"``, ``"int4"``).
        tolerance: Maximum allowed normalized action difference against eager PyTorch.
        max_task_tokens: Maximum token budget for policies supporting dynamic task prompts.
        n_action_steps: Optional Runtime ``ActionChunkTrimmer`` postprocessor step limit.
        seed: Random seed for generating the parity test case.
        trust_remote_code: Explicitly allow exporting policies outside :data:`VERIFIED_EXPORT_POLICIES`.
    """

    task: str = ""
    dataset: str | None = None
    dtype: Literal["float32", "bfloat16", "float16"] | None = None
    device: str = "CPU"
    quantize: str | None = None
    tolerance: float | None = None
    max_task_tokens: int | None = None
    n_action_steps: int | None = None
    seed: int = 1000
    trust_remote_code: bool = False


_DEFAULT_ENV = training_worker_env()
_SMOLVLA_ENV = training_worker_env("smolvla")
_PI_ENV = training_worker_env("pi")
_MOLMOACT2_ENV = training_worker_env("molmoact2")

_EXPORT_ENV = export_worker_env()
_EXPORT_FAMILY_ENVS: Mapping[str, WorkerEnv] = MappingProxyType(
    {
        "smolvla": export_worker_env("smolvla"),
        "pi": export_worker_env("pi"),
        "molmoact2": export_worker_env("molmoact2"),
        "vla": export_worker_env("vla"),
    },
)


def _export_env_for_policy(policy_type: str | None) -> WorkerEnv:
    family = export_family_for_policy(policy_type)
    if family is not None and family in _EXPORT_FAMILY_ENVS:
        return _EXPORT_FAMILY_ENVS[family]
    return _EXPORT_ENV


class LeRobotEngine(TrainingEngine):
    """Run LeRobot's own trainer and exporter in version-locked worker environments.

    Policy families with their own dependencies (``smolvla``, the ``pi`` family and
    ``molmoact2``) run in their own environment, built from ``envs/default/`` (training)
    or ``envs/export/`` (export) with that family's extra; every other policy uses the
    corresponding project without a family extra.
    """

    framework: ClassVar[str] = "lerobot"
    env: ClassVar[WorkerEnv] = _DEFAULT_ENV
    policy_envs: ClassVar[Mapping[str, WorkerEnv]] = MappingProxyType(
        {
            "smolvla": _SMOLVLA_ENV,
            "pi0": _PI_ENV,
            "pi05": _PI_ENV,
            "pi0_fast": _PI_ENV,
            "molmoact2": _MOLMOACT2_ENV,
        },
    )
    stock_export_formats: ClassVar[tuple[str, ...]] = ("native",)
    enablements: ClassVar[tuple[Enablement, ...]] = (
        Enablement(
            name="executorch-openvino",
            provides={policy: {"export": ("executorch_openvino",)} for policy in sorted(VERIFIED_EXPORT_POLICIES)},
            upstream=LEROBOT_EXPORT_UPSTREAM_REF,
            env=_EXPORT_ENV,
        ),
        Enablement(
            name="openvino",
            provides={policy: {"export": ("openvino",)} for policy in sorted(VERIFIED_EXPORT_POLICIES)},
            upstream="https://github.com/huggingface/lerobot (pending upstream PR)",
            env=_EXPORT_ENV,
        ),
    )

    def __init__(self, *, interpreter: str | Path | None = None) -> None:
        """Create a LeRobot engine with a managed or explicitly supplied interpreter."""
        self._interpreter = Path(interpreter).expanduser() if interpreter is not None else None

    def train(self, config: TrainPipelineConfig, hooks: TrainHooks | None = None) -> Run:
        """Serialize a LeRobot config and execute LeRobot's native training CLI.

        The host and worker must use the same LeRobot version when exchanging a
        native config object. Call :meth:`cli` with native flags or a config file
        when that version cannot be matched.

        Returns:
            Native run output and checkpoint paths.

        Raises:
            FileNotFoundError: If config serialization does not write ``train_config.json``.
            TypeError: If *config* is not a native serializable LeRobot config.
            ValueError: If *config* trains a reward model or an unsupported policy, or the native
                device is unsupported by the worker environment.
        """
        save_pretrained = getattr(config, "save_pretrained", None)
        if not callable(save_pretrained):
            msg = "LeRobotEngine.train() requires a native TrainPipelineConfig with save_pretrained()."
            raise TypeError(msg)
        if getattr(config, "reward_model", None) is not None:
            raise ValueError(_REWARD_MODEL_UNSUPPORTED)

        policy = getattr(config, "policy", None)
        policy_type = require_supported_policy(getattr(policy, "type", None))
        env = self.environment_for_policy(policy_type)
        expected_version = _host_lerobot_version(env)
        configured_device = getattr(policy, "device", None)
        device, worker_accelerator = _resolve_worker_device(configured_device, allowed=env.accelerators)
        with tempfile.TemporaryDirectory(prefix="physicalai-lerobot-config-") as temporary:
            config_dir = Path(temporary)
            save_pretrained(config_dir)
            config_path = config_dir / "train_config.json"
            if not config_path.is_file():
                msg = "LeRobot TrainPipelineConfig.save_pretrained() did not write train_config.json."
                raise FileNotFoundError(msg)
            return run_worker(
                env,
                [f"--config_path={config_path}"],
                device=device,
                hooks=hooks,
                accelerator=worker_accelerator,
                interpreter=self._interpreter,
                expected_version=expected_version,
            )

    @classmethod
    def cli(
        cls,
        argv: Sequence[str],
        *,
        interpreter: str | Path | None = None,
    ) -> int:
        """Run ``lerobot-train`` with inherited standard streams and its exact exit code.

        In a managed environment the command runs through the plugin's worker script, which
        rejects an unsupported policy or reward model once LeRobot has resolved its command
        line and config file. A user-managed *interpreter* runs ``lerobot-train`` as given.

        Returns:
            The native command's exit status.

        Raises:
            RuntimeError: If the engine does not declare a worker environment.
            ValueError: If the command line sets a reward model or an unsupported policy, or the
                native device has no matching worker extra.
        """
        if interpreter is not None:
            # A user-managed environment: run its LeRobot as given, without choosing a managed
            # environment (no Hub lookups) or checking the device against managed profiles.
            python = Path(interpreter).expanduser()
            script_name = "lerobot-train.exe" if os.name == "nt" else "lerobot-train"
            return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # Native CLI passthrough; arguments are never shell-evaluated.  # nosec B603
                [str(scripts_dir(python) / script_name), *argv],
                shell=False,
            )
        if cls.env is None:
            msg = "LeRobotEngine requires a worker environment."
            raise RuntimeError(msg)
        if any(argument.startswith("--reward_model.") for argument in argv):
            raise ValueError(_REWARD_MODEL_UNSUPPORTED)
        policy_type = _policy_type_from_cli(argv)
        if policy_type is not None or not _asks_for_help(argv):
            require_supported_policy(policy_type)
        env = cls.environment_for_policy(policy_type)
        requested_device = _device_from_cli(argv)
        _, accelerator = _resolve_worker_device(requested_device, allowed=env.accelerators)
        python = interpreter_for(env, accelerator=accelerator, interpreter=None)
        # The host only reads the command line to choose an environment. The worker script checks
        # what LeRobot resolves (a config file's `policy.path`, repeated flags, a Hub revision)
        # before training. `-P` keeps the plugin folder off the worker's sys.path.
        with resources.as_file(env.worker) as worker:
            return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # Native CLI passthrough; arguments are never shell-evaluated.  # nosec B603
                [str(python), "-P", str(worker), *argv],
                env={**os.environ, _NATIVE_CLI_VARIABLE: "1"},
                shell=False,
            )

    @classmethod
    def run(
        cls,
        settings: Mapping[str, Any],
        *,
        dataset: Path,
        output_dir: Path,
        device: str,
        hooks: TrainHooks | None = None,
        interpreter: str | Path | None = None,
    ) -> Run:
        """Run headless training with Studio-owned dataset/output/device overrides.

        Returns:
            Native run output and checkpoint paths.

        Raises:
            RuntimeError: If the engine does not declare a worker environment.
            ValueError: If the settings set a reward model or an unsupported policy, or the
                selected native device is unsupported.
        """
        if cls.env is None:
            msg = "LeRobotEngine requires a worker environment."
            raise RuntimeError(msg)
        # Flatten first so nested ({"policy": {"type": ...}}) and dotted ({"policy.type": ...})
        # settings select the same environment and device.
        flattened: dict[str, object] = {}
        _flatten_settings(settings, "", flattened)
        if any(
            value is not None and (name == "reward_model" or name.startswith("reward_model."))
            for name, value in flattened.items()
        ):
            raise ValueError(_REWARD_MODEL_UNSUPPORTED)
        env = cls.environment_for_policy(require_supported_policy(_settings_policy_type(flattened)))
        settings_device = _settings_device(flattened)
        requested_device = (
            settings_device if device.strip().lower() == "auto" and settings_device is not None else device
        )
        resolved_device, accelerator = _resolve_worker_device(requested_device, allowed=env.accelerators)
        flattened.update(
            {
                "dataset.repo_id": "local",
                "dataset.root": str(dataset),
                "output_dir": str(output_dir),
                "policy.device": resolved_device,
            },
        )
        flags = [f"--{name}={_native_value(value)}" for name, value in flattened.items()]
        return run_worker(
            env,
            flags,
            device=resolved_device,
            hooks=hooks,
            accelerator=accelerator,
            interpreter=interpreter,
        )

    @classmethod
    def export(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        backend: str | None = "native",
        *,
        options: LeRobotExportOptions | None = None,
        interpreter: str | Path | None = None,
    ) -> Path:
        """Export or package a LeRobot policy for Physical AI Runtime inference.

        Supported backends:
        - ``"native"`` (default): packages the native PyTorch checkpoint (``.lerobot``).
        - ``"openvino"``: compiles via isolated ``lerobot-export`` to direct OpenVINO IR
          (``.xml`` + ``.bin``) inside a dual-compatible ``.lerobot-bundle`` Runtime package.
        - ``"executorch_openvino"``: compiles via isolated ``lerobot-export`` to ExecuTorch
          ``.pte`` with ``OpenvinoPartitioner`` inside a ``.lerobot-bundle`` Runtime package.

        Args:
            source: Trained ``Run``, local checkpoint directory, or Hugging Face Hub ``repo_id``.
            output_path: Destination directory for the Runtime package (must not exist yet).
            backend: Target export backend (``"native"``, ``"openvino"``, or ``"executorch_openvino"``).
            options: Optional :class:`LeRobotExportOptions` for ``openvino`` / ``executorch_openvino``.
            interpreter: Optional user-managed Python interpreter bypassing ``envs/export/``.

        Returns:
            The completed Runtime package directory.

        Raises:
            FileExistsError: If *output_path* already exists.
            ValueError: If *backend* or policy type is unsupported.
        """
        selected_backend = "native" if backend is None else backend
        if selected_backend not in _EXPORT_BACKENDS:
            supported = ", ".join(sorted(_EXPORT_BACKENDS))
            msg = f"{cls.framework} supports export backends ({supported}), got {backend!r}."
            raise ValueError(msg)

        if selected_backend == "native":
            from .package import package_lerobot_checkpoint  # ruff: ignore[import-outside-top-level]

            return package_lerobot_checkpoint(source, Path(output_path))

        return cls._export_bundle(
            source,
            Path(output_path),
            backend=selected_backend,
            options=options or LeRobotExportOptions(),
            interpreter=interpreter,
        )

    @classmethod
    def _export_bundle(
        cls,
        source: Run | str | Path,
        output_path: Path,
        *,
        backend: str,
        options: LeRobotExportOptions,
        interpreter: str | Path | None,
    ) -> Path:
        from .package import package_lerobot_bundle  # ruff: ignore[import-outside-top-level]

        policy_source, policy_type = _resolve_export_source(source)
        validated_policy = require_exportable_policy(policy_type, trust_remote_code=options.trust_remote_code)

        output_candidate = output_path.expanduser()
        if output_candidate.exists() or output_candidate.is_symlink():
            msg = f"Export directory already exists: {output_candidate}"
            raise FileExistsError(msg)
        package_root = output_candidate.resolve()
        package_root.parent.mkdir(parents=True, exist_ok=True)

        export_env = _export_env_for_policy(validated_policy)
        with tempfile.TemporaryDirectory(prefix=f".{package_root.name}-export-", dir=package_root.parent) as staging:
            staged_bundle_dir = Path(staging) / f"{validated_policy}.lerobot-bundle"
            export_args = _build_export_cli_args(
                policy_source=policy_source,
                output_dir=staged_bundle_dir,
                backend=backend,
                options=options,
            )
            if backend == "openvino":
                with resources.as_file(openvino_enablement_resource()) as ov_dir:
                    run_worker(
                        export_env,
                        export_args,
                        device="cpu",
                        accelerator="cpu",
                        interpreter=interpreter,
                        extra_env={_ENABLEMENT_OV_DIR_ENV: str(ov_dir)},
                    )
            else:
                run_worker(
                    export_env,
                    export_args,
                    device="cpu",
                    accelerator="cpu",
                    interpreter=interpreter,
                )

            worker_env_key = _EXPORT_ENV.key() if backend == "executorch_openvino" else None
            return package_lerobot_bundle(
                staged_bundle_dir,
                package_root,
                policy_name=validated_policy,
                export_backend=backend,
                worker_env_key=worker_env_key,
                n_action_steps=options.n_action_steps,
                trust_remote_code=options.trust_remote_code,
            )

    @classmethod
    def policies(cls) -> list[str]:
        """List the LeRobot policy types this plugin supports.

        Returns:
            Supported policy identifiers that LeRobot's installed config registry provides.
        """
        import_module("lerobot.policies")
        from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]

        choices = PreTrainedConfig.get_known_choices()
        names = choices.keys() if isinstance(choices, Mapping) else choices
        return sorted(policy for policy in names if isinstance(policy, str) and policy in SUPPORTED_POLICY_TYPES)

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Return LeRobot's native training schema specialized for one policy.

        Raises:
            ValueError: If *policy* is invalid or not enabled by this plugin.
        """
        if not isinstance(policy, str) or _POLICY_NAME_RE.fullmatch(policy) is None:
            msg = f"Invalid LeRobot policy name {policy!r}."
            raise ValueError(msg)
        if policy not in cls.policies():
            msg = f"LeRobot policy {policy!r} is not enabled by this plugin."
            raise ValueError(msg)

        from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]
        from lerobot.configs.train import TrainPipelineConfig  # ruff: ignore[import-outside-top-level]

        policy_config = PreTrainedConfig.get_choice_class(policy)
        schema = _dataclass_json_schema(TrainPipelineConfig)
        schema["properties"]["policy"] = _dataclass_json_schema(policy_config)
        return schema


def _resolve_export_source(source: Run | str | Path) -> tuple[str, str | None]:
    """Resolve an export source (Run, local path, or HF Hub repo_id) to ``(policy_path, policy_type)``."""
    from .package import _native_model_directory, _policy_name  # ruff: ignore[import-outside-top-level]

    if isinstance(source, str) and not Path(source).expanduser().exists() and _HUB_REPO_RE.fullmatch(source):
        return source, _hub_policy_type(source)

    native_dir = _native_model_directory(source)
    return str(native_dir), _policy_name(native_dir)


def _build_export_cli_args(
    *,
    policy_source: str,
    output_dir: Path,
    backend: str,
    options: LeRobotExportOptions,
) -> list[str]:
    args = [
        f"--policy.path={policy_source}",
        f"--backend.type={backend}",
        f"--output_dir={output_dir}",
        f"--seed={options.seed}",
    ]
    if options.task:
        args.append(f"--task={options.task}")
    if options.dataset:
        dataset_path = Path(options.dataset).expanduser()
        if dataset_path.exists():
            args.append("--dataset.repo_id=local")
            args.append(f"--dataset.root={dataset_path.resolve()}")
        else:
            args.append(f"--dataset.repo_id={options.dataset}")
    if options.dtype is not None:
        args.append(f"--policy.dtype={options.dtype}")
    if options.quantize is not None:
        args.append(f"--quantize={options.quantize}")
    if options.tolerance is not None:
        args.append(f"--tolerance={options.tolerance}")
    if options.max_task_tokens is not None:
        args.append(f"--max_task_tokens={options.max_task_tokens}")
    if backend == "openvino" and options.device:
        args.append(f"--backend.ov_device={options.device}")
    return args


def _dataclass_json_schema(config_type: type) -> dict[str, Any]:
    """Return the JSON schema of a native LeRobot config dataclass.

    LeRobot configs have ``torch.dtype`` fields (``policy.dtype`` and the FSDP mixed-precision
    dtypes), which pydantic cannot describe. The CLI takes them as dtype names such as
    ``bfloat16``, so the schema describes them as strings. Arbitrary types are allowed only while
    this schema is built: the config is set on the class and removed again, and nested
    dataclasses inherit it.

    Returns:
        The JSON schema of *config_type*.
    """
    import torch  # ruff: ignore[import-outside-top-level]  # Already loaded by LeRobot's config modules.
    from pydantic import ConfigDict, TypeAdapter  # ruff: ignore[import-outside-top-level]
    from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue  # ruff: ignore[import-outside-top-level]
    from pydantic_core import core_schema  # ruff: ignore[import-outside-top-level]

    class _NativeConfigSchema(GenerateJsonSchema):
        def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
            if schema.get("cls") is torch.dtype:
                return {"type": "string", "description": "Torch dtype name, for example `bfloat16`."}
            return super().is_instance_schema(schema)

    had_config = "__pydantic_config__" in vars(config_type)
    previous = vars(config_type).get("__pydantic_config__")
    config_type.__pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[attr-defined]
    try:
        return TypeAdapter(config_type).json_schema(schema_generator=_NativeConfigSchema)
    finally:
        if had_config:
            config_type.__pydantic_config__ = previous  # type: ignore[attr-defined]
        else:
            delattr(config_type, "__pydantic_config__")


def settings_to_flags(settings: Mapping[str, Any]) -> list[str]:
    """Convert nested native settings to LeRobot dotted CLI flags.

    Returns:
        One native ``--field=value`` token per setting.
    """
    flattened: dict[str, object] = {}
    _flatten_settings(settings, "", flattened)
    return [f"--{name}={_native_value(value)}" for name, value in flattened.items()]


def _flatten_settings(
    settings: Mapping[str, Any],
    prefix: str,
    output: dict[str, object],
    *,
    depth: int = 0,
) -> None:
    if depth > _MAX_SETTING_DEPTH:
        msg = f"LeRobot settings exceed the maximum nesting depth of {_MAX_SETTING_DEPTH}."
        raise ValueError(msg)
    for raw_name, value in settings.items():
        if not isinstance(raw_name, str) or _SETTING_NAME_RE.fullmatch(raw_name) is None:
            msg = f"Invalid LeRobot setting name {raw_name!r}."
            raise ValueError(msg)
        name = f"{prefix}.{raw_name}" if prefix else raw_name
        if isinstance(value, Mapping):
            _flatten_settings(value, name, output, depth=depth + 1)
        else:
            output[name] = value


def _native_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _resolve_worker_device(device: object, *, allowed: tuple[str, ...]) -> tuple[str, str]:
    if not isinstance(device, str) or not device.strip() or device.strip().lower() == "auto":
        accelerator = resolve_accelerator("auto", allowed=allowed)
        return _device_for_accelerator(accelerator), accelerator

    accelerator = _accelerator_for_device(device)
    selected = resolve_accelerator(accelerator, allowed=allowed)
    return device, selected


def _accelerator_for_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "xpu" or normalized.startswith("xpu:"):
        return "xpu"
    if normalized == "cuda" or normalized.startswith("cuda:"):
        return "cuda"
    msg = f"Unsupported LeRobot device {device!r}; choose cpu, cuda, xpu, or auto."
    raise ValueError(msg)


def _device_for_accelerator(accelerator: str) -> str:
    return accelerator


def _settings_device(flattened: Mapping[str, object]) -> str | None:
    device = flattened.get("policy.device")
    return device if isinstance(device, str) and device.strip() else None


_CLI_OPTIONS = ("--policy.device", "--policy.type", "--policy.path", "--config_path")


def _cli_options(arguments: Sequence[str]) -> dict[str, str]:
    """Collect the native options the engine reads, in ``--name=value`` or ``--name value`` form.

    Returns:
        The last value given for each option in :data:`_CLI_OPTIONS`.
    """
    found: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        for option in _CLI_OPTIONS:
            if argument.startswith(f"{option}="):
                found[option] = argument.partition("=")[2]
                break
            if argument == option and index + 1 < len(arguments):
                index += 1
                found[option] = arguments[index]
                break
        index += 1
    return found


def _asks_for_help(arguments: Sequence[str]) -> bool:
    return any(argument in {"-h", "--help"} for argument in arguments)


def _device_from_cli(arguments: Sequence[str]) -> str | None:
    options = _cli_options(arguments)
    configured_device = options.get("--policy.device")
    if configured_device is not None:
        return configured_device
    return _device_from_config(options.get("--policy.path"), policy_config=True) or _device_from_config(
        options.get("--config_path"),
    )


def _policy_type_from_cli(arguments: Sequence[str]) -> str | None:
    """Find the native policy type a ``lerobot-train`` command line will train.

    Precedence follows LeRobot's ``TrainPipelineConfig._resolve_pretrained_from_cli()``:
    ``--policy.path`` (local or on the Hugging Face Hub) overwrites ``self.policy`` from its
    ``config.json``, then ``--policy.type``, then ``--config_path``.

    Returns:
        The policy type, or ``None`` when the command line does not determine it.
    """
    options = _cli_options(arguments)
    policy_path = options.get("--policy.path")
    if policy_path:
        return _policy_type_from_config(policy_path, policy_config=True) or _hub_policy_type(policy_path)
    policy_type = options.get("--policy.type")
    if policy_type:
        return policy_type
    return _policy_type_from_config(options.get("--config_path"))


def _settings_policy_type(flattened: Mapping[str, object]) -> str | None:
    """Find the policy type in flattened settings, with the same precedence as the native CLI.

    Returns:
        The policy type, or ``None`` when the settings do not determine it.
    """
    path = flattened.get("policy.path")
    if isinstance(path, str) and path.strip():
        return _policy_type_from_config(path, policy_config=True) or _hub_policy_type(path)
    policy_type = flattened.get("policy.type")
    if isinstance(policy_type, str) and policy_type.strip():
        return policy_type.strip()
    config_path = flattened.get("config_path")
    return _policy_type_from_config(config_path) if isinstance(config_path, str) and config_path.strip() else None


def _policy_type_from_config(config_path: str | None, *, policy_config: bool = False) -> str | None:
    """Read the policy type from a local policy ``config.json`` or a training config.

    Returns:
        The policy type, or ``None`` when *config_path* is not a readable local config.
    """
    if config_path is None:
        return None
    path = Path(config_path).expanduser()
    if path.is_dir():
        names = ("config.json", "train_config.json") if policy_config else ("train_config.json", "config.json")
        candidates = tuple(path / name for name in names)
    elif path.is_file():
        candidates = (path,)
    else:
        return None
    for candidate in candidates:
        data = _load_native_config(candidate)
        if not isinstance(data, Mapping):
            continue
        policy = data.get("policy")
        policy_type = policy.get("type") if isinstance(policy, Mapping) else data.get("type")
        if isinstance(policy_type, str) and policy_type.strip():
            return policy_type.strip()
    return None


def _hub_policy_type(repo_id: str) -> str | None:
    """Read the policy type from a Hugging Face Hub checkpoint's ``config.json``.

    Returns:
        The policy type, or ``None`` when *repo_id* is not a Hub id or its config cannot be read.
        The engine then falls back to its default environment.
    """
    if _HUB_REPO_RE.fullmatch(repo_id) is None:
        return None
    try:
        from huggingface_hub import hf_hub_download  # ruff: ignore[import-outside-top-level]

        config_file = hf_hub_download(repo_id, "config.json")  # nosec B615  # Reads metadata only; the worker loads the model.
    except Exception as exc:  # ruff: ignore[blind-except]  # Any download failure only means the default environment.
        logger.warning(
            "Could not read the policy type of %s (%s); using the default LeRobot environment.",
            repo_id,
            exc,
        )
        return None
    return _policy_type_from_config(config_file, policy_config=True)


def _device_from_config(config_path: str | None, *, policy_config: bool = False) -> str | None:
    if config_path is None:
        return None
    path = Path(config_path).expanduser()
    if path.is_dir():
        names = ("config.json", "train_config.json") if policy_config else ("train_config.json", "config.json")
        candidates = tuple(path / name for name in names)
    elif path.is_file():
        candidates = (path,)
    else:
        return None

    for candidate in candidates:
        data = _load_native_config(candidate)
        if not isinstance(data, Mapping):
            continue
        policy = data.get("policy")
        if isinstance(policy, Mapping):
            device = policy.get("device")
            if isinstance(device, str) and device.strip():
                return device
        device = data.get("device")
        if isinstance(device, str) and device.strip():
            return device
    return None


def _load_native_config(path: Path) -> object:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415  # LeRobot accepts YAML native configs.
        except ImportError:
            return None
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _host_lerobot_version(env: WorkerEnv) -> Version:
    try:
        host_version = Version(metadata.version(env.package))
    except (metadata.PackageNotFoundError, ValueError) as exc:
        msg = "LeRobot config objects require the host LeRobot package to be installed."
        raise RuntimeError(msg) from exc
    required_version = locked_version(env)
    supported = supported_versions(env)
    if supported is not None and host_version not in supported:
        msg = f"Host LeRobot version {host_version} is outside the plugin's supported range {supported}."
        raise RuntimeError(msg)
    if host_version != required_version:
        msg = (
            f"Host LeRobot version {host_version} does not match the worker lock's version "
            f"{required_version}; use native CLI arguments or install matching versions."
        )
        raise RuntimeError(msg)
    return host_version


__all__ = ["LeRobotEngine", "LeRobotExportOptions", "settings_to_flags"]

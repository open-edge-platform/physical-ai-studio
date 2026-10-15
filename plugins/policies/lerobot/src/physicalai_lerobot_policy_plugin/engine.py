# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native LeRobot engine, isolated from the Studio host environment."""

from __future__ import annotations

import json
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Executes the framework's installed native script with shell=False.
import tempfile
from collections.abc import Mapping, Sequence
from importlib import import_module, metadata, resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from packaging.version import Version
from physicalai.engine import (
    Engine,
    Run,
    TrainHooks,
    WorkerEnv,
    interpreter_for,
    locked_version,
    resolve_accelerator,
    run_worker,
    scripts_dir,
    supported_versions,
)

from .policy_support import UNREVIEWED_POLICY_TYPES

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

_POLICY_NAME_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")
_SETTING_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*")
_MAX_SETTING_DEPTH = 10
_PLUGIN_DISTRIBUTION = "physicalai-lerobot-policy-plugin"
_PLUGIN_ROOT = resources.files(__package__)


class LeRobotEngine(Engine):
    """Run LeRobot's own trainer in its version-locked worker environment."""

    framework: ClassVar[str] = "lerobot"
    env: ClassVar[WorkerEnv] = WorkerEnv(
        framework=framework,
        project=_PLUGIN_ROOT / "env",
        worker=_PLUGIN_ROOT / "_worker.py",
        distribution=_PLUGIN_DISTRIBUTION,
        package="lerobot",
    )
    stock_export_formats: ClassVar[tuple[str, ...]] = ("native",)

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
            ValueError: If the native device is unsupported by the worker environment.
        """
        save_pretrained = getattr(config, "save_pretrained", None)
        if not callable(save_pretrained):
            msg = "LeRobotEngine.train() requires a native TrainPipelineConfig with save_pretrained()."
            raise TypeError(msg)

        expected_version = _host_lerobot_version(self.env)
        policy = getattr(config, "policy", None)
        configured_device = getattr(policy, "device", None)
        device, worker_accelerator = _resolve_worker_device(configured_device, allowed=self.env.accelerators)
        with tempfile.TemporaryDirectory(prefix="physicalai-lerobot-config-") as temporary:
            config_dir = Path(temporary)
            save_pretrained(config_dir)
            config_path = config_dir / "train_config.json"
            if not config_path.is_file():
                msg = "LeRobot TrainPipelineConfig.save_pretrained() did not write train_config.json."
                raise FileNotFoundError(msg)
            return run_worker(
                self.env,
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

        Returns:
            The native command's exit status.

        Raises:
            RuntimeError: If the engine does not declare a worker environment.
            ValueError: If the native device has no matching worker extra.
        """
        if cls.env is None:
            msg = "LeRobotEngine requires a worker environment."
            raise RuntimeError(msg)
        requested_device = _device_from_cli(argv)
        _, accelerator = _resolve_worker_device(requested_device, allowed=cls.env.accelerators)
        python = interpreter_for(cls.env, accelerator=accelerator, interpreter=interpreter)
        script_name = "lerobot-train.exe" if os.name == "nt" else "lerobot-train"
        executable = scripts_dir(python) / script_name
        return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # Native CLI passthrough; arguments are never shell-evaluated.
            [str(executable), *argv],
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
            ValueError: If the selected native device is unsupported.
        """
        if cls.env is None:
            msg = "LeRobotEngine requires a worker environment."
            raise RuntimeError(msg)
        settings_device = _settings_device(settings)
        requested_device = settings_device if device.strip().lower() == "auto" and settings_device is not None else device
        resolved_device, accelerator = _resolve_worker_device(requested_device, allowed=cls.env.accelerators)
        flattened: dict[str, object] = {}
        _flatten_settings(settings, "", flattened)
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
            cls.env,
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
    ) -> Path:
        """Package a native LeRobot checkpoint for Runtime's adapter.

        Returns:
            The Runtime package directory.

        Raises:
            ValueError: If *backend* is not the native package format.
        """
        if backend not in {None, "native"}:
            msg = f"{cls.framework} supports only the native export format, got {backend!r}."
            raise ValueError(msg)
        from .package import package_lerobot_checkpoint  # ruff: ignore[import-outside-top-level]

        return package_lerobot_checkpoint(source, Path(output_path))

    @classmethod
    def policies(cls) -> list[str]:
        """List built-in LeRobot policy types safe for the stock engine.

        Returns:
            Policy identifiers supported by LeRobot's installed config registry.
        """
        import_module("lerobot.policies")
        from lerobot.configs.policies import PreTrainedConfig  # ruff: ignore[import-outside-top-level]

        choices = PreTrainedConfig.get_known_choices()
        names = choices.keys() if isinstance(choices, Mapping) else choices
        return sorted(policy for policy in names if isinstance(policy, str) and policy not in UNREVIEWED_POLICY_TYPES)

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
        from pydantic import TypeAdapter  # ruff: ignore[import-outside-top-level]

        policy_config = PreTrainedConfig.get_choice_class(policy)
        schema = TypeAdapter(TrainPipelineConfig).json_schema()
        schema["properties"]["policy"] = TypeAdapter(policy_config).json_schema()
        return schema


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
        return "cu128"
    msg = f"Unsupported LeRobot device {device!r}; choose cpu, cuda, xpu, or auto."
    raise ValueError(msg)


def _device_for_accelerator(accelerator: str) -> str:
    return "cuda" if accelerator == "cu128" else accelerator


def _settings_device(settings: Mapping[str, Any]) -> str | None:
    policy = settings.get("policy")
    if not isinstance(policy, Mapping):
        return None
    device = policy.get("device")
    return device if isinstance(device, str) and device.strip() else None


def _device_from_cli(arguments: Sequence[str]) -> str | None:
    configured_device: str | None = None
    config_path: str | None = None
    policy_path: str | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        options = (("--policy.device", "device"), ("--config_path", "config"), ("--policy.path", "policy"))
        for option, destination in options:
            if argument.startswith(f"{option}="):
                value = argument.partition("=")[2]
                if destination == "device":
                    configured_device = value
                elif destination == "config":
                    config_path = value
                else:
                    policy_path = value
                break
            if argument == option and index + 1 < len(arguments):
                index += 1
                value = arguments[index]
                if destination == "device":
                    configured_device = value
                elif destination == "config":
                    config_path = value
                else:
                    policy_path = value
                break
        index += 1
    if configured_device is not None:
        return configured_device
    return _device_from_config(policy_path, policy_config=True) or _device_from_config(config_path)


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
    if host_version not in supported:
        msg = f"Host LeRobot version {host_version} is outside the plugin's supported range {supported}."
        raise RuntimeError(msg)
    if host_version != required_version:
        msg = (
            f"Host LeRobot version {host_version} does not match the worker lock's version "
            f"{required_version}; use native CLI arguments or install matching versions."
        )
        raise RuntimeError(msg)
    return host_version


__all__ = ["LeRobotEngine", "settings_to_flags"]

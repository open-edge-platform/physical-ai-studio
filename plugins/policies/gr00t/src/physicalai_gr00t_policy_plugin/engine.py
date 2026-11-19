# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Isaac-GR00T N1.7 engine for Physical AI.

GR00T trains with its own upstream fine-tuning script in the pinned ``default``
environment. Native checkpoints package for the plugin's Runtime adapter. The
Robotics AI Suite enablement adds OpenVINO export in its own environment: the
same upstream commit with the Suite's verified patch series applied.
"""

from __future__ import annotations

import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Native CLI passthrough; shell is disabled.
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from physicalai.engine import Enablement, Run, TrainHooks, TrainingEngine, WorkerEnv, run_worker

from . import environments
from .config import Gr00tTrainConfig
from .environments import FORMAT_NATIVE, FORMAT_OPENVINO, FRAMEWORK
from .package import POLICY, IoSpec, read_run_info, write_native_package, write_openvino_package

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

BACKBONE_SOURCE_ENV = "PHYSICALAI_GR00T_BACKBONE_SOURCE"
_EXPORT_RECORD = "physicalai_export.json"
_RUN_INFO = "physicalai_run.json"


def _worker_env(declaration: environments.WorkerEnvironmentSpec) -> WorkerEnv:
    """Adapt a Runtime isolate declaration to the optional Engine API.

    Returns:
        The equivalent Engine worker declaration.
    """
    return WorkerEnv(
        framework=declaration.framework,
        name=declaration.name,
        project=declaration.source,
        worker=declaration.worker,
        distribution=declaration.distribution,
        package=declaration.package,
        accelerators=declaration.accelerators,
        python=declaration.profile.python,
        extras=dict.fromkeys(declaration.accelerators, declaration.profile.extras),
    )


NATIVE = _worker_env(environments.NATIVE)
OPENVINO = _worker_env(environments.OPENVINO)
OPENVINO_ENABLEMENT = Enablement(
    name=OPENVINO.name,
    provides={POLICY: {"export": (FORMAT_OPENVINO,)}},
    upstream=(
        f"{environments.UPSTREAM.repository}@{environments.UPSTREAM.revision} + "
        f"{environments.ROBOTICS_AI_SUITE_OPENVINO.repository}@"
        f"{environments.ROBOTICS_AI_SUITE_OPENVINO.revision}:"
        f"{environments.ROBOTICS_AI_SUITE_OPENVINO.directory}"
    ),
    env=OPENVINO,
)


class Gr00tEngine(TrainingEngine):
    """Train, package, and export NVIDIA Isaac-GR00T N1.7 policies."""

    framework: ClassVar[str] = FRAMEWORK
    env: ClassVar[WorkerEnv | None] = NATIVE
    training_accelerators: ClassVar[tuple[str, ...]] = ("cu128",)
    stock_export_formats: ClassVar[tuple[str, ...]] = (FORMAT_NATIVE,)
    enablements: ClassVar[tuple[Enablement, ...]] = (OPENVINO_ENABLEMENT,)

    def __init__(
        self,
        *,
        interpreter: str | Path | None = None,
        project: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """Create an engine.

        Args:
            interpreter: User-managed interpreter for the ``default`` environment.
            project: Isaac-GR00T checkout used by *interpreter*.
            device: ``auto``, ``cuda``, or ``cuda:N``; GR00T fine-tuning requires CUDA.
        """
        self._interpreter = interpreter
        self._project = project
        self._env = _device_env(device)

    def train(self, config: Gr00tTrainConfig | Sequence[str], hooks: TrainHooks | None = None) -> Run:
        """Fine-tune with upstream ``launch_finetune.py`` in the ``default`` environment.

        Args:
            config: Typed settings, or native ``launch_finetune.py`` argument tokens.
            hooks: Optional progress and cancellation hooks.

        Returns:
            The native output directory and final checkpoint.
        """
        variables = dict(self._env)
        if isinstance(config, Gr00tTrainConfig):
            argv = config.to_argv()
            if config.backbone_source:
                variables[BACKBONE_SOURCE_ENV] = config.backbone_source
        else:
            argv = [str(token) for token in config]
        if self._interpreter is not None:
            _, project, _ = environments.resolve(
                environments.NATIVE,
                interpreter=self._interpreter,
                project=self._project,
            )
            variables[environments.PROJECT_ENV] = str(project)
        run = run_worker(
            NATIVE,
            argv,
            device="cuda",
            hooks=hooks,
            interpreter=self._interpreter,
            worker=environments.TRAIN_WORKER,
            extra_env=variables,
        )
        if run.checkpoint is not None:
            _record_environment(run.checkpoint, environments.NATIVE)
        return run

    @classmethod
    def cli(cls, argv: Sequence[str], *, interpreter: str | Path | None = None) -> int:
        """Run upstream ``launch_finetune.py`` with inherited streams and exit status.

        Returns:
            The native command's exit status.
        """
        python, project, _ = environments.resolve(environments.NATIVE, interpreter=interpreter)
        with resources.as_file(environments.TRAIN_WORKER) as worker:
            return subprocess.call(  # ruff: ignore[subprocess-without-shell-equals-true]  # argv list; shell disabled.
                [str(python), str(worker), *argv],
                env={**os.environ, environments.PROJECT_ENV: str(project)},
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
        """Run headless fine-tuning with Studio-owned dataset, output, and device.

        Returns:
            The native output directory and final checkpoint.
        """
        config = Gr00tTrainConfig.from_settings(settings, dataset=dataset, output_dir=output_dir)
        return cls(interpreter=interpreter, device=device).train(config, hooks)

    @classmethod
    def export(cls, source: Run | str | Path, output_path: str | Path, backend: str | None = FORMAT_NATIVE) -> Path:
        """Package a native checkpoint for Runtime, natively or through OpenVINO.

        Returns:
            The Runtime package directory.

        Raises:
            ValueError: If *backend* is not ``native`` or ``openvino``.
        """
        if backend in {None, FORMAT_NATIVE}:
            return cls.export_native(source, output_path)
        if backend == FORMAT_OPENVINO:
            return cls.export_openvino(source, output_path)
        msg = f"GR00T exports 'native' or 'openvino', got {backend!r}."
        raise ValueError(msg)

    @classmethod
    def export_native(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        *,
        dataset_path: str | Path | None = None,
        embodiment_tag: str | None = None,
    ) -> Path:
        """Package the native checkpoint for PyTorch inference through Runtime.

        Args:
            source: Native run or checkpoint directory, including upstream-released checkpoints.
            output_path: New Runtime package directory.
            dataset_path: Dataset whose ``meta/modality.json`` defines the I/O mapping;
                defaults to the training dataset.
            embodiment_tag: Required for checkpoints not trained by this plugin.

        Returns:
            The Runtime package directory.
        """
        checkpoint = _checkpoint(source)
        run_info = _run_info(checkpoint, embodiment_tag)
        return write_native_package(
            checkpoint,
            Path(output_path),
            io=_io_spec(run_info, dataset_path),
            run_info=run_info,
            environment=environments.provenance(environments.NATIVE),
        )

    @classmethod
    def export_openvino(  # Mirrors the Suite exporter's options.
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        *,
        precision: str = "fp16",
        llm_lang_tokens: int | None = 64,
        dataset_path: str | Path | None = None,
        embodiment_tag: str | None = None,
        interpreter: str | Path | None = None,
        project: str | Path | None = None,
    ) -> Path:
        """Export with the Robotics AI Suite single-IR OpenVINO path.

        Args:
            source: Native run or checkpoint directory, including upstream-released checkpoints.
            output_path: New Runtime package directory.
            precision: ``fp16`` (Suite default) or ``fp32`` IR weights.
            llm_lang_tokens: Language tokens reserved in the static LLM sequence.
            dataset_path: Dataset that supplies representative capture inputs and
                the I/O mapping; defaults to the training dataset.
            embodiment_tag: Required for checkpoints not trained by this plugin.
            interpreter: User-managed interpreter for the patched environment.
            project: Patched checkout used by *interpreter*.

        Returns:
            The Runtime package directory.
        """
        checkpoint = _checkpoint(source)
        run_info = _run_info(checkpoint, embodiment_tag)
        io = _io_spec(run_info, dataset_path)
        sample = dataset_path or run_info.get("dataset_path")
        variables: dict[str, str] = {}
        if interpreter is not None:
            _, checkout, _ = environments.resolve(
                environments.OPENVINO,
                interpreter=interpreter,
                project=project,
            )
            variables[environments.PROJECT_ENV] = str(checkout)

        def export(ov_dir: Path) -> dict[str, Any]:
            request = {
                "model_path": str(checkpoint),
                "dataset_path": str(sample) if sample else None,
                "embodiment_tag": run_info["embodiment_tag"],
                "output_dir": str(ov_dir),
                "precision": precision,
                "llm_lang_tokens": llm_lang_tokens,
                "device": "cpu",
            }
            run_worker(
                OPENVINO,
                [json.dumps(request)],
                device="cpu",
                interpreter=interpreter,
                worker=environments.EXPORT_WORKER,
                extra_env=variables,
            )
            record_path = ov_dir / _EXPORT_RECORD
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record_path.unlink()
            return record

        return write_openvino_package(
            export,
            Path(output_path),
            checkpoint=checkpoint,
            io=io,
            run_info=run_info,
            environment=environments.provenance(environments.OPENVINO),
        )

    @classmethod
    def policies(cls) -> list[str]:
        """Return the policy identifiers this engine provides.

        Returns:
            ``["gr00t_n1d7"]``.
        """
        return [POLICY]

    @classmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Return the JSON schema of the native fine-tuning settings.

        Returns:
            JSON schema for :class:`Gr00tTrainConfig`.

        Raises:
            ValueError: If *policy* is not provided by this engine.
        """
        if policy != POLICY:
            msg = f"Unknown {FRAMEWORK} policy {policy!r}."
            raise ValueError(msg)
        return Gr00tTrainConfig.json_schema()


def _device_env(device: str) -> dict[str, str]:
    normalized = device.strip().lower()
    if normalized in {"auto", "cuda", "gpu"}:
        return {}
    if normalized.startswith("cuda:") and normalized.removeprefix("cuda:").isdigit():
        return {"CUDA_VISIBLE_DEVICES": normalized.removeprefix("cuda:")}
    msg = f"GR00T fine-tuning requires CUDA; device {device!r} is not supported."
    raise ValueError(msg)


def _record_environment(checkpoint: Path, env: environments.WorkerEnvironmentSpec) -> None:
    path = checkpoint / _RUN_INFO
    if not path.is_file():
        return
    info = json.loads(path.read_text(encoding="utf-8"))
    info["environment"] = environments.provenance(env)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def _checkpoint(source: Run | str | Path) -> Path:
    if isinstance(source, Run):
        if source.framework != FRAMEWORK or source.checkpoint is None:
            msg = "Export needs a GR00T run with a native checkpoint."
            raise ValueError(msg)
        path = source.checkpoint
    else:
        path = Path(source)
    checkpoint = Path(path).expanduser().resolve(strict=True)
    if not (checkpoint / "config.json").is_file():
        msg = f"No GR00T checkpoint (config.json) in {checkpoint}."
        raise FileNotFoundError(msg)
    return checkpoint


def _run_info(checkpoint: Path, embodiment_tag: str | None) -> dict[str, Any]:
    run_info = read_run_info(checkpoint)
    if embodiment_tag is not None:
        run_info["embodiment_tag"] = embodiment_tag
    if not run_info.get("embodiment_tag"):
        msg = "Pass embodiment_tag for checkpoints that were not trained by this plugin."
        raise ValueError(msg)
    if "backbone_source" not in run_info:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        run_info["backbone_source"] = config.get("model_name")
    return run_info


def _io_spec(run_info: Mapping[str, Any], dataset_path: str | Path | None) -> IoSpec:
    dataset = dataset_path or run_info.get("dataset_path")
    if not dataset:
        msg = "Pass dataset_path so the package can record the dataset's modality mapping."
        raise ValueError(msg)
    modality = Path(dataset).expanduser() / "meta" / "modality.json"
    if not modality.is_file():
        msg = f"Dataset modality file not found: {modality}"
        raise FileNotFoundError(msg)
    return IoSpec.from_modality(modality)

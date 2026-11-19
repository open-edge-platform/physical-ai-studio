# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral APIs for framework and model-repository engines.

Engine classes are discovered lazily from the ``physicalai.engines`` entry
point group. Every engine packages models for Runtime; :class:`TrainingEngine`
subclasses also train. Worker environments are declared as :class:`WorkerEnv`
values built from a bundled project or a pinned Git checkout with optional
patch series that are bundled with the plugin or published elsewhere. Importing this module does not import Lightning,
LeRobot, PyTorch, or Runtime inference adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, overload

from . import _registry
from ._environment import (
    PreparedEnvironment,
    engine_cache_root,
    interpreter_for,
    list_prepared,
    locked_version,
    prepare,
    prepare_environment,
    prune,
    prune_prepared,
    resolve_accelerator,
    scripts_dir,
    supported_versions,
    to_isolate_environment,
)
from ._runner import run_worker
from ._sources import (
    BundledPatches,
    BundledProject,
    GitProject,
    PatchFile,
    PatchSeries,
    ProjectSource,
    PublishedPatches,
)
from ._types import Capabilities, Enablement, Run, TrainHooks, TrainingCancelledError, WorkerEnv

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from physicalai.inference.model import InferenceModel
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    from physicalai.train.trainer import Trainer


class Engine(ABC):
    """Common contract for one framework or model repository.

    Every engine packages models for Runtime and reports what it provides.
    Engines that also train subclass :class:`TrainingEngine`. Capabilities that
    need different framework code, such as a patched export path, are declared
    as :class:`Enablement` entries, each optionally with its own environment.
    """

    framework: ClassVar[str]
    env: ClassVar[WorkerEnv | None] = None
    enablements: ClassVar[tuple[Enablement, ...]] = ()
    stock_export_formats: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def export(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        backend: str | None = None,
    ) -> Path:
        """Package a native checkpoint for Runtime inference."""

    @classmethod
    @abstractmethod
    def policies(cls) -> list[str]:
        """List policy identifiers this engine provides."""

    @classmethod
    def environments(cls) -> tuple[WorkerEnv, ...]:
        """Return the engine's own environment followed by enablement environments.

        Returns:
            Declared worker environments.

        Raises:
            ValueError: If two environments share a name.
        """
        declared = [cls.env] if cls.env is not None else []
        declared.extend(enablement.env for enablement in cls.enablements if enablement.env is not None)
        names = [env.name for env in declared]
        if len(names) != len(set(names)):
            msg = f"Engine {cls.framework!r} declares duplicate environment names: {names}."
            raise ValueError(msg)
        return tuple(declared)

    @classmethod
    def environment(cls, name: str | None = None) -> WorkerEnv:
        """Return the engine's environment, or the environment of enablement *name*.

        Returns:
            The selected worker environment.

        Raises:
            ValueError: If the engine or enablement declares no such environment.
        """
        if name is None:
            if cls.env is None:
                msg = f"Engine {cls.framework!r} does not use a managed worker environment."
                raise ValueError(msg)
            return cls.env
        for enablement in cls.enablements:
            if enablement.name == name and enablement.env is not None:
                return enablement.env
        available = ", ".join(enablement.name for enablement in cls.enablements if enablement.env is not None)
        msg = f"Engine {cls.framework!r} has no enablement environment {name!r}. Available: {available or '<none>'}."
        raise ValueError(msg)

    @classmethod
    def capabilities(cls, policy: str) -> Capabilities:
        """Return stock and validated enablement capabilities for *policy*.

        Training accelerators are reported only by :class:`TrainingEngine` subclasses.

        Raises:
            ValueError: If *policy* is not provided by this engine.
        """
        if policy not in cls.policies():
            msg = f"Unknown {cls.framework} policy {policy!r}."
            raise ValueError(msg)

        trains = issubclass(cls, TrainingEngine)
        accelerators: set[str] = set()
        if trains:
            accelerators.update(getattr(cls, "training_accelerators", ()))
            if cls.env is not None:
                accelerators.update(cls.env.accelerators)
        export_formats = set(cls.stock_export_formats)
        active_enablements: list[str] = []
        for enablement in cls.enablements:
            provides = enablement.provides.get(policy)
            if provides is None:
                continue
            if trains:
                accelerators.update(provides.get("train", ()))
            export_formats.update(provides.get("export", ()))
            active_enablements.append(enablement.name)
        return Capabilities(
            training_accelerators=tuple(sorted(accelerators)),
            export_formats=tuple(sorted(export_formats)),
            enablements=tuple(sorted(active_enablements)),
        )

    @overload
    @classmethod
    def create(
        cls,
        name: Literal["physicalai"],
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> Trainer: ...

    @overload
    @classmethod
    def create(
        cls,
        name: Literal["lerobot"],
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> LeRobotEngine: ...

    @overload
    @classmethod
    def create(
        cls,
        name: str,
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> Engine: ...

    @classmethod
    def create(cls, name: str, /, **options: Any) -> Engine:
        """Instantiate the engine registered as *name*.

        Returns:
            An instance of the selected engine. Prefer its concrete class for static option checking.
        """
        engine_class = cls.get(name)
        return engine_class(**options)

    @overload
    @classmethod
    def get(cls, name: Literal["physicalai"], /) -> type[Trainer]: ...

    @overload
    @classmethod
    def get(cls, name: Literal["lerobot"], /) -> type[LeRobotEngine]: ...

    @overload
    @classmethod
    def get(cls, name: str, /) -> type[Engine]: ...

    @classmethod
    def get(cls, name: str, /) -> type[Engine]:
        """Return the engine class registered as *name*.

        Returns:
            The class registered under *name*.
        """
        return _registry.get_engine(name, Engine)

    @classmethod
    def names(cls) -> list[str]:
        """List installed engine identifiers without importing engine modules.

        Returns:
            Installed engine identifiers in sorted order.
        """
        return _registry.names()

    @staticmethod
    def load(package: str | Path, **kwargs: Any) -> InferenceModel:  # ruff: ignore[any-type]  # Runtime accepts adapter options.
        """Load a Runtime policy package through the Runtime API.

        Returns:
            The initialized Runtime inference model.
        """
        from physicalai.inference.model import InferenceModel as RuntimeInferenceModel  # ruff: ignore[import-outside-top-level]

        return RuntimeInferenceModel(package, **kwargs)


class TrainingEngine(Engine):
    """An engine that also trains through its framework's native recipe."""

    training_accelerators: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def cli(
        cls,
        argv: Sequence[str],
        *,
        interpreter: str | Path | None = None,
    ) -> int:
        """Run the framework's native training CLI and return its exit code."""

    @classmethod
    @abstractmethod
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
        """Run a headless training job from native settings and standard inputs."""

    @classmethod
    @abstractmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Return the native training config schema for *policy*."""


__all__ = [
    "BundledPatches",
    "BundledProject",
    "Capabilities",
    "Enablement",
    "Engine",
    "GitProject",
    "PatchFile",
    "PatchSeries",
    "PreparedEnvironment",
    "ProjectSource",
    "PublishedPatches",
    "Run",
    "TrainHooks",
    "TrainingCancelledError",
    "TrainingEngine",
    "WorkerEnv",
    "engine_cache_root",
    "interpreter_for",
    "list_prepared",
    "locked_version",
    "prepare",
    "prepare_environment",
    "prune",
    "prune_prepared",
    "resolve_accelerator",
    "run_worker",
    "scripts_dir",
    "supported_versions",
    "to_isolate_environment",
]

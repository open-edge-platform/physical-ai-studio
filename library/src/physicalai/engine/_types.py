# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight shared types for framework engines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable
    from pathlib import Path

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from ._sources import ProjectSource


class TrainingCancelledError(RuntimeError):
    """Raised when a training hook requests cancellation."""

    def __init__(self, step: int | None = None) -> None:
        self.step = step
        detail = f" after step {step}" if step is not None else ""
        super().__init__(f"Training cancelled{detail}.")


class TrainHooks(Protocol):
    """Progress and cancellation hooks shared by all native trainers."""

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        """Report one completed optimization step."""

    def should_stop(self) -> bool:
        """Return whether the current training run should stop."""


@dataclass(frozen=True)
class Run:
    """Result of a native framework training invocation."""

    framework: str
    output_dir: Path | None
    checkpoint: Path | None


@dataclass(frozen=True)
class Capabilities:
    """Validated training and export capabilities for one policy."""

    training_accelerators: tuple[str, ...] = ()
    export_formats: tuple[str, ...] = ()
    enablements: tuple[str, ...] = ()


@dataclass(frozen=True)
class Enablement:
    """Validated capabilities that an engine adds beyond its stock framework.

    An enablement that needs different framework code or dependencies declares
    its own :class:`WorkerEnv` in ``env``; for example, an upstream checkout with
    a verified downstream patch series. Capabilities are listed only after they
    pass validation.

    Attributes:
        name: Enablement identifier, unique within its engine.
        provides: Per-policy operations and formats, such as ``{"act": {"export": ("openvino",)}}``.
        upstream: Human-readable reference of the code providing the capability.
        env: Worker environment providing the capability, if it differs from the engine's.
    """

    name: str
    provides: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    upstream: str | None = None
    env: WorkerEnv | None = None


@dataclass(frozen=True)
class WorkerEnv:
    """Locked, isolated Python project used by an engine worker.

    ``project`` is a :class:`~physicalai.engine.ProjectSource`, or a directory
    containing ``pyproject.toml`` and ``uv.lock`` (shorthand for
    :class:`~physicalai.engine.BundledProject`). ``worker`` is the default
    standard-library-compatible script executed by the environment's interpreter.

    Attributes:
        framework: Engine identifier that owns the environment.
        project: Source of the uv project.
        worker: Default worker script.
        distribution: Plugin distribution that declares the environment.
        package: Framework distribution installed in the environment; workers report its version.
        accelerators: Accelerator profiles the environment supports.
        name: Environment name, unique within its framework. Enablements use their own names.
        python: Python version requested from uv, or ``None`` to follow the project.
        extras: uv extras per accelerator profile. Defaults to the profile's own name.
    """

    framework: str
    project: Path | Traversable | ProjectSource
    worker: Path | Traversable
    distribution: str
    package: str
    accelerators: tuple[str, ...] = ("cpu", "cu128", "xpu")
    name: str = "default"
    python: str | None = None
    extras: Mapping[str, tuple[str, ...]] | None = field(default=None, hash=False)

    @property
    def source(self) -> ProjectSource:
        """The project source, wrapping a plain directory as a bundled project."""
        from ._sources import BundledProject, ProjectSource  # ruff: ignore[import-outside-top-level]  # Avoid an import cycle.

        if isinstance(self.project, ProjectSource):
            return self.project
        return BundledProject(self.project)

    def extras_for(self, accelerator: str) -> tuple[str, ...]:
        """Return the uv extras that select *accelerator*.

        Returns:
            Extra names passed to ``uv sync``.
        """
        if self.extras is None:
            return (accelerator,)
        return tuple(self.extras.get(accelerator, ()))

    def key(self) -> str:
        """Hash every input that determines the environment's contents.

        Returns:
            Hexadecimal SHA-256 key.
        """
        identity = {
            "format": 2,
            "framework": self.framework,
            "name": self.name,
            "python": self.python,
            "extras": {accelerator: list(self.extras_for(accelerator)) for accelerator in sorted(self.accelerators)},
            "source": self.source.identity(),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def supported(self) -> SpecifierSet | None:
        """Return the package range declared by the plugin, if any."""
        from ._environment import supported_versions  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        return supported_versions(self)

    def pinned(self) -> Version:
        """Return the exact package version from a bundled worker lock."""
        from ._environment import locked_version  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        return locked_version(self)

    def lock_hash(self) -> str:
        """Return the SHA-256 digest of a bundled worker lock."""
        from ._environment import lock_hash  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        return lock_hash(self)


__all__ = [
    "Capabilities",
    "Enablement",
    "Run",
    "TrainHooks",
    "TrainingCancelledError",
    "WorkerEnv",
]

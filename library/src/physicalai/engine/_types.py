# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight shared types for framework engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable
    from pathlib import Path

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version


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
    """Declarative capabilities supplied by a validated engine enablement.

    ``upstream`` records the pinned upstream release or commit that provides
    the capability. Patch retrieval and application are intentionally not part
    of this type. Enablement environments are ordinary locked ``WorkerEnv``
    projects; an enablement is listed only after its behavior has passed validation.
    """

    name: str
    provides: Mapping[str, Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    upstream: str | None = None


@dataclass(frozen=True)
class WorkerEnv:
    """Locked, isolated Python project used by an engine worker.

    ``project`` contains ``pyproject.toml`` and ``uv.lock``. ``worker`` is a
    standard-library-compatible script executed by the managed or supplied
    interpreter. The supported package range comes from the plugin's installed
    distribution metadata; the exact version comes from the worker lock.
    """

    framework: str
    project: Path | Traversable
    worker: Path | Traversable
    distribution: str
    package: str
    accelerators: tuple[str, ...] = ("cpu", "cu128", "xpu")

    def supported(self) -> SpecifierSet:
        """Return the supported package range from plugin metadata."""
        from ._environment import supported_versions  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        return supported_versions(self)

    def pinned(self) -> Version:
        """Return the exact package version from the worker lock."""
        from ._environment import locked_version  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        return locked_version(self)

    def lock_hash(self) -> str:
        """Return the SHA-256 digest of the worker lock."""
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

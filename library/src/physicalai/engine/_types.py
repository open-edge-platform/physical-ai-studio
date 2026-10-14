# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight public types for framework integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class TrainingCancelledError(RuntimeError):
    """Raised when a training hook requests cancellation."""

    def __init__(self, step: int | None = None) -> None:
        self.step = step
        detail = f" after step {step}" if step is not None else ""
        super().__init__(f"Training cancelled{detail}.")


class TrainHooks(Protocol):
    """Progress and cancellation hooks supplied to a framework trainer."""

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        """Report one completed optimization step."""

    def should_stop(self) -> bool:
        """Return whether the current training run should stop."""


class Framework(Protocol):
    """Contract implemented by an installed training framework."""

    name: str

    def accepts(self, config: object) -> bool:
        """Return whether this framework recognizes *config*."""

    def train(self, config: object | Sequence[str], hooks: TrainHooks) -> Run:
        """Train with a native config object or native CLI arguments."""

    def export(
        self,
        source: Run | str | Path,
        output_dir: Path,
        format: str | None = None,  # noqa: A002
    ) -> Path:
        """Package a native checkpoint for Runtime inference."""


@dataclass(frozen=True)
class Run:
    """Result of a framework training invocation.

    Attributes:
        framework: Framework identifier that produced the run.
        output_dir: Native framework output directory when the framework reports one.
        checkpoint: Native inference checkpoint when the run produced one.
    """

    framework: str
    output_dir: Path | None
    checkpoint: Path | None

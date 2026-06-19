# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training backend contract shared by local and remote implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from schemas import Job, Model, Snapshot
    from schemas.job import TrainJobPayload


class TrainingCanceledError(Exception):
    """Raised by a backend when training stops because cancellation was requested.

    Distinct from a genuine failure: the worker marks the job CANCELED and logs
    at info level instead of dumping an error traceback.
    """


class ProgressReporter(Protocol):
    """Report training progress back to the job store and event stream."""

    def __call__(self, progress: int, *, message: str | None = None, extra_info: dict | None = None) -> None:
        """Publish a progress update.

        Args:
            progress: Completion percentage in the range 0-100.
            message: Optional human-readable status line.
            extra_info: Optional telemetry (e.g. step loss). Never used for control flow.
        """
        ...


@dataclass
class TrainingContext:
    """Inputs a backend needs to produce a trained, exported model.

    A backend must leave a fully populated model directory at ``output_dir``
    (checkpoint, logger output, and ``exports/``), report progress through
    ``progress``, and stop promptly when ``should_stop`` returns True.
    """

    job: Job
    model: Model
    snapshot: Snapshot
    payload: TrainJobPayload
    base_model: Model | None
    output_dir: Path
    cache_dir: Path
    progress: ProgressReporter
    should_stop: Callable[[], bool]


@runtime_checkable
class TrainingBackend(Protocol):
    """Strategy that trains a model described by a `TrainingContext`."""

    async def train(self, context: TrainingContext) -> None:
        """Train and export the model, raising on failure."""
        ...

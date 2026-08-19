# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Phase-windowed progress reporting shared by remote training backends.

A remote SSH job has more phases than a local or direct-URL run (connect,
image pull, image verification, trainer start) layered on top of the existing
upload -> train -> download. This module extends the existing progress model
(``ProgressReporter(progress, message, extra_info)``) rather than changing its
contract: each phase owns a slice of the 0-100 bar, and a structured
descriptor rides inside the existing ``extra_info["phase"]`` dict so the UI
can render a stepper while the plain bar still drives unaware consumers.

Local and direct-URL backends are untouched by this module: they keep their
own ``SNAPSHOT_UPLOAD_PROGRESS`` / ``TRAINING_PROGRESS_END`` windows in
``services.training_backends.remote`` and never attach a ``phase`` descriptor.
This table is selected only by the (not-yet-implemented) SSH backend.

``PHASE_TABLE_VERSION`` and ``PhaseKey`` must stay in lockstep with whatever
the UI consumes, so both sides drift together rather than silently apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from services.training_backends.base import ProgressReporter

# Bump whenever the set or order of PhaseKey members changes in a way that
# would break a UI consumer keyed off `extra_info["phase"]["version"]`.
PHASE_TABLE_VERSION = 1


class PhaseKey(StrEnum):
    """Ordered stages of an SSH-provisioned training job."""

    CONNECT = "connect"
    IMAGE_PULL = "image_pull"
    IMAGE_VERIFY = "image_verify"
    TRAINER_START = "trainer_start"
    UPLOAD = "upload"
    TRAIN = "train"
    DOWNLOAD = "download"


class PhaseState(StrEnum):
    """Rendering state of one phase in the stepper."""

    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"
    WAITING = "waiting"
    FAILED = "failed"


@dataclass(frozen=True)
class PhaseWindow:
    """One phase's slice of the overall 0-100 progress bar."""

    key: PhaseKey
    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.end <= 100:
            raise ValueError(f"Invalid phase window for {self.key}: [{self.start}, {self.end}]")


# The SSH phase table for an SSH-provisioned training job. Local and direct-URL backends keep
# their own SNAPSHOT_UPLOAD_PROGRESS/TRAINING_PROGRESS_END windows unchanged;
# retuning those shared constants to fit this table would shift their curves
# and force rewriting their existing progress assertions for no gain.
SSH_PHASE_WINDOWS: tuple[PhaseWindow, ...] = (
    PhaseWindow(PhaseKey.CONNECT, "Connect & preflight", 0, 2),
    PhaseWindow(PhaseKey.IMAGE_PULL, "Image pull", 2, 5),
    PhaseWindow(PhaseKey.IMAGE_VERIFY, "Image verification", 5, 7),
    PhaseWindow(PhaseKey.TRAINER_START, "Trainer start", 7, 9),
    PhaseWindow(PhaseKey.UPLOAD, "Dataset upload", 9, 17),
    PhaseWindow(PhaseKey.TRAIN, "Training", 17, 96),
    PhaseWindow(PhaseKey.DOWNLOAD, "Model download", 96, 100),
)


def _map_into_window(sub_progress: float, window: PhaseWindow) -> int:
    """Map a phase-local 0-100 percentage into its reserved window."""
    clamped = max(0.0, min(100.0, sub_progress))
    span = window.end - window.start
    return window.start + round(clamped / 100 * span)


def report_phase(
    progress: ProgressReporter,
    windows: Sequence[PhaseWindow],
    key: PhaseKey,
    *,
    state: PhaseState = PhaseState.ACTIVE,
    sub_progress: float | None = 0.0,
    message: str | None = None,
    extra_info: dict | None = None,
) -> None:
    """Report progress for one phase, attaching a structured stepper descriptor.

    Maps ``sub_progress`` (0-100, the phase's own completion) into the phase's
    reserved slice of the overall bar and calls ``progress`` with a
    ``phase`` descriptor in ``extra_info`` alongside any caller-supplied keys.
    ``sub_progress=None`` marks the phase indeterminate (e.g. a Docker pull
    with no stable percentage): the bar pins to the window start so a spinner
    can be shown instead of a misleading exact percent.

    Args:
        progress: The backend's `ProgressReporter` callback.
        windows: The ordered phase table this job uses.
        key: Which phase is being reported.
        state: The phase's rendering state for the stepper.
        sub_progress: The phase's own 0-100 completion, or None if indeterminate.
        message: Optional human-readable status line.
        extra_info: Additional telemetry to merge alongside the phase descriptor.

    Raises:
        ValueError: If `key` is not present in `windows`.
    """
    index = next((i for i, window in enumerate(windows) if window.key == key), None)
    if index is None:
        raise ValueError(f"Phase {key} is not part of the given phase table")
    window = windows[index]
    indeterminate = sub_progress is None
    overall_progress = window.start if sub_progress is None else _map_into_window(sub_progress, window)

    descriptor: dict = dict(extra_info or {})
    descriptor["phase"] = {
        "version": PHASE_TABLE_VERSION,
        "key": window.key.value,
        "label": window.label,
        "index": index,
        "total": len(windows),
        "state": state.value,
        "indeterminate": indeterminate,
    }
    progress(overall_progress, message=message, extra_info=descriptor)

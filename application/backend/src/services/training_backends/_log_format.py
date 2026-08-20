# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared formatting for training-progress log lines.

Local and remote backends must emit identical job-log lines so the UI log
stream reads the same regardless of where training ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def format_training_progress(*, global_step: int, max_steps: int, loss: float | None) -> str:
    """Build the job-log line for one training step.

    Args:
        global_step: Lightning global step.
        max_steps: Configured maximum steps (floored at 1 to avoid divide-by-zero).
        loss: Step training loss, or None when unavailable.

    Returns:
        A single log line, identical across local and remote backends.
    """
    max_steps = max(1, max_steps)
    progress = min(100, round(global_step / max_steps * 100))
    return f"Training progress: step={global_step}/{max_steps} ({progress}%), train/loss_step={loss}"


def format_validation_start(*, global_step: int, max_steps: int) -> str:
    """Build the job-log line marking the start of validation.

    Args:
        global_step: Lightning global step when validation began.
        max_steps: Configured maximum steps (floored at 1).

    Returns:
        A single log line, identical across local and remote backends.
    """
    return f"Validation started at step={global_step}/{max(1, max_steps)}"


def format_validation_progress(*, batch: int, loss: float | None) -> str:
    """Build the job-log line for one validation batch.

    Args:
        batch: 1-based validation batch index.
        loss: Step validation loss, or None when unavailable.

    Returns:
        A single log line, identical across local and remote backends.
    """
    return f"Validation progress: batch={batch}, val/loss_step={loss}"


def format_validation_summary(*, global_step: int, val_loss: float | None, elapsed_s: float) -> str:
    """Build the job-log line summarizing a finished validation pass.

    Args:
        global_step: Lightning global step when validation finished.
        val_loss: Aggregated validation loss, or None when unavailable.
        elapsed_s: Wall-clock seconds the validation pass took.

    Returns:
        A single log line, identical across local and remote backends.
    """
    return f"Validation finished at step={global_step}, val/loss={val_loss}, elapsed={elapsed_s:.1f}s"


def render_progress_log(extra_info: Mapping[str, Any]) -> str | None:
    """Render a job-log line from progress telemetry, or None when not loggable.

    Consumes the ``extra_info`` schema emitted by
    ``physicalai.train.ProgressReportingCallback`` so local and remote backends
    turn identical telemetry into identical log lines. Fields may arrive from
    remote JSON, so they are coerced defensively; a malformed or non-cadence
    payload yields None instead of raising.

    Args:
        extra_info: Telemetry attached to a progress update.

    Returns:
        A formatted log line, or None when the payload carries nothing to log.
    """
    event = extra_info.get("val_event")
    if event is not None:
        return _render_validation_log(str(event), extra_info)
    return _render_training_log(extra_info)


def _render_training_log(extra_info: Mapping[str, Any]) -> str | None:
    """Render the training-progress line from cadence fields, or None."""
    try:
        global_step = int(extra_info["global_step"])
        max_steps = int(extra_info["max_steps"])
    except (KeyError, TypeError, ValueError):
        # Non-cadence batches carry only the step loss; nothing to log.
        return None
    loss = extra_info.get("train/loss_step")
    loss_val = float(loss) if isinstance(loss, int | float) else None
    return format_training_progress(global_step=global_step, max_steps=max_steps, loss=loss_val)


def _render_validation_log(event: str, extra_info: Mapping[str, Any]) -> str | None:
    """Render the validation line for a start/batch/end event, or None."""
    if event == "start":
        return _render_validation_start(extra_info)
    if event == "batch":
        return _render_validation_batch(extra_info)
    if event == "end":
        return _render_validation_end(extra_info)
    return None


def _render_validation_start(extra_info: Mapping[str, Any]) -> str | None:
    """Render the validation-start line, or None when fields are malformed."""
    try:
        global_step = int(extra_info["global_step"])
        max_steps = int(extra_info["max_steps"])
    except (KeyError, TypeError, ValueError):
        return None
    return format_validation_start(global_step=global_step, max_steps=max_steps)


def _render_validation_batch(extra_info: Mapping[str, Any]) -> str | None:
    """Render the validation-batch line, or None when fields are malformed."""
    try:
        batch = int(extra_info["val_batch"])
    except (KeyError, TypeError, ValueError):
        return None
    loss = extra_info.get("val/loss_step")
    loss_val = float(loss) if isinstance(loss, int | float) else None
    return format_validation_progress(batch=batch, loss=loss_val)


def _render_validation_end(extra_info: Mapping[str, Any]) -> str | None:
    """Render the validation-summary line, or None when fields are malformed."""
    try:
        global_step = int(extra_info["global_step"])
    except (KeyError, TypeError, ValueError):
        return None
    loss = extra_info.get("val/loss")
    loss_val = float(loss) if isinstance(loss, int | float) else None
    elapsed = extra_info.get("val_elapsed_s")
    elapsed_val = float(elapsed) if isinstance(elapsed, int | float) else 0.0
    return format_validation_summary(global_step=global_step, val_loss=loss_val, elapsed_s=elapsed_val)

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Formatting for training and validation progress log lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def _format_training_progress(*, global_step: int, max_steps: int, loss: float | None) -> str:
    """Format a training step progress log line."""
    max_steps = max(1, max_steps)
    progress = min(100, round(global_step / max_steps * 100))
    return f"Training progress: step={global_step}/{max_steps} ({progress}%), train/loss_step={loss}"


def _format_validation_start(*, global_step: int, max_steps: int) -> str:
    """Format a validation-start log line."""
    return f"Validation started at step={global_step}/{max(1, max_steps)}"


def _format_validation_progress(*, batch: int, loss: float | None) -> str:
    """Format a validation batch progress log line."""
    return f"Validation progress: batch={batch}, val/loss_step={loss}"


def _format_validation_summary(*, global_step: int, val_loss: float | None, elapsed_s: float) -> str:
    """Format a validation completion summary log line."""
    return f"Validation finished at step={global_step}, val/loss={val_loss}, elapsed={elapsed_s:.1f}s"


def render_progress_log(extra_info: Mapping[str, Any]) -> str | None:
    """Render a progress log line from ``ProgressReportingCallback`` telemetry.

    Args:
        extra_info: Telemetry payload attached to a progress callback event.

    Returns:
        A formatted log line, or None when the payload is not a loggable cadence
        or validation event.
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
    return _format_training_progress(global_step=global_step, max_steps=max_steps, loss=loss_val)


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
    return _format_validation_start(global_step=global_step, max_steps=max_steps)


def _render_validation_batch(extra_info: Mapping[str, Any]) -> str | None:
    """Render the validation-batch line, or None when fields are malformed."""
    try:
        batch = int(extra_info["val_batch"])
    except (KeyError, TypeError, ValueError):
        return None
    loss = extra_info.get("val/loss_step")
    loss_val = float(loss) if isinstance(loss, int | float) else None
    return _format_validation_progress(batch=batch, loss=loss_val)


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
    return _format_validation_summary(global_step=global_step, val_loss=loss_val, elapsed_s=elapsed_val)

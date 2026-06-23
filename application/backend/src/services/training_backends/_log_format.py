# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared formatting for training-progress log lines.

Local and remote backends must emit identical job-log lines so the UI log
stream reads the same regardless of where training ran.
"""

from __future__ import annotations


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

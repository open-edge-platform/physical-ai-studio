# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Learning rate schedulers for training policies."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def cosine_decay_with_warmup_scheduler(
    optimizer: Optimizer,
    *,
    peak_lr: float,
    decay_lr: float,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int | None = None,
) -> LambdaLR:
    """Create a cosine decay scheduler with linear warmup.

    Matches the schedule used by Physical Intelligence for Pi0/Pi05 training:
    - Linear warmup from ``peak_lr / (num_warmup_steps + 1)`` to ``peak_lr``.
    - Cosine decay from ``peak_lr`` down to ``decay_lr``.
    - After ``num_decay_steps`` the LR stays at ``decay_lr``.

    When ``num_training_steps`` is provided and is less than ``num_decay_steps``,
    both warmup and decay steps are automatically scaled down so the full
    schedule fits within the available training budget.

    Args:
        optimizer: Wrapped optimizer.
        peak_lr: Peak learning rate (reached at end of warmup).
        decay_lr: Final learning rate after cosine decay.
        num_warmup_steps: Number of linear warmup steps.
        num_decay_steps: Total decay horizon (cosine half-period).
        num_training_steps: Total number of training steps. When provided
            and less than ``num_decay_steps``, warmup and decay are scaled
            proportionally to fit.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    actual_warmup_steps = num_warmup_steps
    actual_decay_steps = num_decay_steps

    if num_training_steps is not None and num_training_steps < num_decay_steps:
        scale_factor = num_training_steps / num_decay_steps
        actual_warmup_steps = int(num_warmup_steps * scale_factor)
        actual_decay_steps = num_training_steps
        logger.warning(
            "Auto-scaling LR scheduler: "
            "num_training_steps (%d) < num_decay_steps (%d). "
            "Scaling warmup: %d -> %d, "
            "decay: %d -> %d "
            "(scale factor: %.3f)",
            num_training_steps,
            num_decay_steps,
            num_warmup_steps,
            actual_warmup_steps,
            num_decay_steps,
            actual_decay_steps,
            scale_factor,
        )

    alpha = decay_lr / peak_lr

    def lr_lambda(current_step: int) -> float:
        if current_step < actual_warmup_steps:
            if current_step <= 0:
                return 1.0 / (actual_warmup_steps + 1)
            frac = 1.0 - current_step / actual_warmup_steps
            return (1.0 / (actual_warmup_steps + 1) - 1.0) * frac + 1.0
        step = min(current_step, actual_decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * step / actual_decay_steps))
        return (1.0 - alpha) * cosine_decay + alpha

    return LambdaLR(optimizer, lr_lambda)

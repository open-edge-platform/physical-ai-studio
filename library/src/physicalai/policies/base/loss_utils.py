# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss-related helpers shared across Model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.data.observation import EXTRA

if TYPE_CHECKING:
    import torch


def in_episode_bound(batch: dict[str, Any], exempt_idx: torch.Tensor | None = None) -> torch.Tensor | None:
    """Build the mask of action steps that carry real supervision.

    LeRobot clamps action-chunk queries at episode boundaries, repeating the
    terminal action to fill the chunk and flagging the clamped steps as
    ``action_is_pad``.  Those steps must not supervise the policy, otherwise
    the tail of every episode trains towards a frozen pose.

    Args:
        batch: Preprocessed batch dict, optionally containing
            ``extra.action_is_pad`` as a ``(batch, chunk)`` bool tensor.
        exempt_idx: Optional indices of samples that should stay fully
            weighted despite the padding, for objectives that do not regress
            onto the dataset action (e.g. self-distillation).

    Returns:
        A ``(batch, chunk)`` bool mask, ``True`` where the step should
        contribute to the loss, or ``None`` when the batch carries no
        padding information (e.g. non-chunked datasets).
    """
    actions_is_pad = batch.get(EXTRA + ".action_is_pad")
    if actions_is_pad is None:
        return None
    bound = ~actions_is_pad
    if exempt_idx is not None:
        bound = bound.clone()
        bound[exempt_idx] = True
    return bound


def reduce_losses(losses: torch.Tensor, in_episode_bound: torch.Tensor | None) -> torch.Tensor:
    """Reduce per-element losses to a scalar, ignoring padded action steps.

    Args:
        losses: Per-element losses shaped ``(batch, chunk, action_dim)``.
            Padded steps are zeroed here if they were not already.
        in_episode_bound: Optional ``(batch, chunk)`` bool mask from
            :func:`in_episode_bound`.

    Returns:
        Scalar loss averaged over the valid elements only.
    """
    if in_episode_bound is None:
        return losses.mean()
    masked = losses * in_episode_bound.unsqueeze(-1)
    num_valid = (in_episode_bound.sum() * losses.shape[-1]).clamp_min(1)
    return masked.sum() / num_valid

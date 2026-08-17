# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tensor plumbing shared by the XR1 preprocessor, model and policy.

The reference implementation hardcodes a 60-dimensional dual-arm state and action
layout (arm joints, gripper positions, end-effector poses and rotation matrices).
Nothing here assumes that layout: vectors are padded to the configured
``max_state_dim`` / ``max_action_dim`` and a mask records which entries are real,
so datasets of any width train without touching the architecture.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812


def pad_vector(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad the last dimension of a vector up to ``target_dim``.

    Args:
        vector: Tensor whose last dimension is at most ``target_dim``.
        target_dim: Desired size of the last dimension.

    Returns:
        The padded tensor, or the input unchanged when already wide enough.

    Raises:
        ValueError: If the vector is wider than ``target_dim``, which would
            silently drop action dimensions.
    """
    current = vector.shape[-1]
    if current > target_dim:
        msg = f"Vector of width {current} exceeds target width {target_dim}"
        raise ValueError(msg)
    if current == target_dim:
        return vector
    return F.pad(vector, (0, target_dim - current))


def build_action_mask(
    action: torch.Tensor,
    valid_dim: int,
    temporal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the mask marking which action entries carry supervision.

    Args:
        action: Padded action tensor of shape ``(batch, horizon, padded_dim)``.
        valid_dim: Number of real (unpadded) action dimensions.
        temporal_mask: Optional per-timestep validity mask of shape
            ``(batch, horizon)``, used when a chunk runs past the end of an
            episode.

    Returns:
        Float mask of the same shape as ``action``.

    Raises:
        ValueError: If ``valid_dim`` exceeds the padded width.
    """
    padded_dim = action.shape[-1]
    if valid_dim > padded_dim:
        msg = f"valid_dim ({valid_dim}) exceeds padded action width ({padded_dim})"
        raise ValueError(msg)

    mask = torch.zeros_like(action)
    mask[..., :valid_dim] = 1.0
    if temporal_mask is not None:
        mask *= temporal_mask[..., None].to(mask.dtype)
    return mask


def resize_with_pad(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize images to ``(height, width)`` preserving aspect ratio, padding the rest.

    Distorting the aspect ratio changes the apparent geometry of the scene, which
    matters for a manipulation policy, so the image is letterboxed instead.

    Args:
        images: Tensor of shape ``(batch, channels, in_height, in_width)``.
        height: Target height.
        width: Target width.
        mode: Interpolation mode passed to :func:`torch.nn.functional.interpolate`.

    Returns:
        Tensor of shape ``(batch, channels, height, width)``.

    Raises:
        ValueError: If ``images`` is not a 4D tensor.
    """
    if images.ndim != 4:  # noqa: PLR2004 - (batch, channels, height, width)
        msg = f"Expected a 4D (batch, channels, height, width) tensor, got shape {tuple(images.shape)}"
        raise ValueError(msg)

    _, _, in_height, in_width = images.shape
    if (in_height, in_width) == (height, width):
        return images

    ratio = min(height / in_height, width / in_width)
    scaled_height = max(1, round(in_height * ratio))
    scaled_width = max(1, round(in_width * ratio))

    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(
        images,
        size=(scaled_height, scaled_width),
        mode=mode,
        align_corners=align_corners,
    )

    pad_height = height - scaled_height
    pad_width = width - scaled_width
    top = pad_height // 2
    left = pad_width // 2
    return F.pad(resized, (left, pad_width - left, top, pad_height - top))


def continue_position_ids(
    vlm_position_ids: torch.Tensor,
    query_length: int,
    *,
    batch_size: int,
    suffix_offset: int = 0,
    suffix_length: int = 0,
) -> torch.Tensor:
    """Continue the backbone's 3D MRoPE grid into the action expert's query.

    The action tokens start one position after the furthest prompt position. The
    non-prefix action positions are pushed a further ``suffix_offset`` places away,
    which is how the reference implementation separates already-executed actions
    from the ones being predicted.

    Args:
        vlm_position_ids: Backbone grid of shape ``(3, batch, seq)``.
        query_length: Length of the action expert's query sequence.
        batch_size: Query batch size.
        suffix_offset: Extra offset applied to the predicted action positions.
        suffix_length: Number of trailing positions treated as predictions.

    Returns:
        Position grid of shape ``(3, batch, query_length)``.
    """
    device = vlm_position_ids.device
    base = torch.arange(query_length, device=device).view(1, 1, -1).repeat(3, batch_size, 1)
    base = base + vlm_position_ids.max(dim=-1).values[..., None] + 1
    if suffix_offset and suffix_length:
        base[:, :, -suffix_length:] += suffix_offset
    return base


def build_dit_attention_mask(
    cache_mask: torch.Tensor,
    query_length: int,
    *,
    prefix_length: int = 0,
    prefix_mask_prob: float = 0.0,
    state_length: int = 1,
    keep_last: int = 2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build the action expert's attention mask over the cache and its own query.

    The query attends to every valid prompt position and causally to itself. During
    training with an action prefix, entries of the prefix are dropped at random so
    the model cannot rely on a complete prefix at inference time.

    Args:
        cache_mask: Padding mask over the cached prompt, shape ``(batch, seq)``.
        query_length: Length of the query sequence (sink + state + actions).
        prefix_length: Number of leading action tokens supplied as a prefix.
        prefix_mask_prob: Probability of dropping an individual prefix token.
        state_length: Number of state tokens in the query.
        keep_last: Number of prefix tokens never dropped.
        generator: Optional RNG for reproducible masking.

    Returns:
        Boolean mask of shape ``(batch, 1, query_length, seq + query_length)``.
    """
    batch_size = cache_mask.shape[0]
    device = cache_mask.device

    expanded_cache = cache_mask[:, None, :].expand(-1, query_length, -1)
    causal = torch.tril(torch.ones(batch_size, query_length, query_length, device=device))

    if prefix_length > keep_last and prefix_mask_prob > 0.0:
        action_start = 1 + state_length
        prefix_end = action_start + prefix_length - keep_last
        suffix_start = action_start + prefix_length
        dropped = torch.rand(prefix_length - keep_last, device=device, generator=generator) < prefix_mask_prob
        causal = causal.clone()
        causal[:, suffix_start:, action_start:prefix_end] *= (~dropped).to(causal.dtype)

    return torch.cat([expanded_cache.to(causal.dtype), causal], dim=-1)[:, None].bool()

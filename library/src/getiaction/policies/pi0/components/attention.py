# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Attention utilities for Pi0/Pi0.5 models.

This module provides attention-related utilities including:
- Attention mask construction utilities

Note: AdaRMSNorm has been moved to gemma.py for consistency with lerobot's
implementation which requires returning (output, gate) tuple for gated residuals.

Based on OpenPI implementation with PyTorch-only support.
"""

from __future__ import annotations

import torch


def make_attention_mask_2d(
    pad_masks: torch.Tensor,
    att_masks: torch.Tensor,
) -> torch.Tensor:
    """Create 2D attention mask from padding and attention masks.

    This implements the attention masking pattern from big_vision/OpenPI:
    - Tokens can attend to valid input tokens with cumulative mask_ar <= their own
    - Supports various attention patterns (causal, prefix-LM, block-causal)

    Attention patterns via att_masks:
        - [1,1,1,1,1,1]: Pure causal attention
        - [0,0,0,1,1,1]: Prefix-LM (first 3 bidirectional, last 3 causal)
        - [1,0,1,0,1,0]: Block-causal attention

    Args:
        pad_masks: Padding mask of shape (batch, seq_len). True for valid tokens.
        att_masks: Attention pattern mask of shape (batch, seq_len).
            1 where previous tokens cannot attend, 0 for shared attention.

    Returns:
        2D attention mask of shape (batch, seq_len, seq_len).
        True where attention is allowed.

    Raises:
        ValueError: If att_masks or pad_masks are not 2D tensors.

    Example:
        >>> pad_masks = torch.ones(2, 10, dtype=torch.bool)
        >>> att_masks = torch.zeros(2, 10)
        >>> att_masks[:, 5:] = 1  # Causal after position 5
        >>> mask_2d = make_attention_mask_2d(pad_masks, att_masks)
    """
    expected_ndim = 2
    if att_masks.ndim != expected_ndim:
        msg = f"att_masks must be 2D, got {att_masks.ndim}D"
        raise ValueError(msg)
    if pad_masks.ndim != expected_ndim:
        msg = f"pad_masks must be 2D, got {pad_masks.ndim}D"
        raise ValueError(msg)

    # Cumulative sum determines attention boundaries
    cumsum = torch.cumsum(att_masks, dim=1)

    # Token i can attend to token j if cumsum[j] <= cumsum[i]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

    # Also mask out padding tokens
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]

    return att_2d_masks & pad_2d_masks


def prepare_4d_attention_mask(
    mask_2d: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    min_value: float = -3.4028235e38,
) -> torch.Tensor:
    """Convert 2D attention mask to 4D format for transformers.

    Args:
        mask_2d: 2D attention mask of shape (batch, seq_len, seq_len).
        dtype: Output dtype for the mask values.
        min_value: Value to use for masked positions (large negative).

    Returns:
        4D attention mask of shape (batch, 1, seq_len, seq_len).
        Uses 0.0 for allowed attention, min_value for masked.
    """
    # Add head dimension: (batch, seq, seq) -> (batch, 1, seq, seq)
    mask_4d = mask_2d[:, None, :, :]

    # Convert to additive mask format
    return torch.where(mask_4d, torch.tensor(0.0, dtype=dtype), torch.tensor(min_value, dtype=dtype))

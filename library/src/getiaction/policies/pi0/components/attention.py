# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Attention utilities for Pi0/Pi0.5 models.

This module provides attention-related utilities including:
- AdaRMSNorm: Adaptive RMSNorm for Pi0.5 timestep conditioning
- Attention mask construction utilities

Based on OpenPI implementation with PyTorch-only support.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for Pi0.5 timestep conditioning.

    Modulates the RMSNorm output based on a conditioning signal (timestep embedding).
    Used in Pi0.5 to inject flow matching timestep information into the action expert.

    Args:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.

    Example:
        >>> norm = AdaRMSNorm(hidden_size=1024)
        >>> x = torch.randn(2, 10, 1024)
        >>> cond = torch.randn(2, 1024)  # timestep embedding
        >>> output = norm(x, cond)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize AdaRMSNorm.

        Args:
            hidden_size: Dimension of the input features.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

        # Conditioning projection: maps conditioning to scale factor
        self.ada_linear = nn.Linear(hidden_size, hidden_size)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            RMS-normalized tensor.
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x *= torch.rsqrt(variance + self.eps)
        return x * self.weight

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Apply adaptive RMSNorm.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size).
            conditioning: Optional conditioning tensor of shape (batch, hidden_size).
                If None, applies standard RMSNorm without adaptation.

        Returns:
            Normalized tensor with optional adaptive scaling.
        """
        # Standard RMSNorm
        output = self._rms_norm(hidden_states.float()).to(hidden_states.dtype)

        # Apply adaptive scaling if conditioning provided
        if conditioning is not None:
            # Project conditioning to scale factor
            scale = self.ada_linear(conditioning)
            # Expand for broadcasting: (batch, hidden) -> (batch, 1, hidden)
            scale = scale.unsqueeze(1)
            output *= 1.0 + scale

        return output


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

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2025 Physical Intelligence
# SPDX-License-Identifier: Apache-2.0

"""Attention utilities for Pi0/Pi0.5 models."""

from __future__ import annotations

import torch
from torch import nn


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for Pi0.5 timestep conditioning."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Initialize AdaRMSNorm layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.ada_linear = nn.Linear(hidden_size, hidden_size)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x *= torch.rsqrt(variance + self.eps)
        return x * self.weight

    def forward(self, hidden_states: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Apply adaptive RMSNorm with optional conditioning."""  # noqa: DOC201
        output = self._rms_norm(hidden_states.float()).to(hidden_states.dtype)
        if conditioning is not None:
            scale = self.ada_linear(conditioning).unsqueeze(1)
            output *= 1.0 + scale
        return output


def make_attention_mask_2d(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Create 2D attention mask from padding and attention masks."""  # noqa: DOC201, DOC501
    expected_ndim = 2
    if att_masks.ndim != expected_ndim:
        msg = f"att_masks must be 2D, got {att_masks.ndim}D"
        raise ValueError(msg)
    if pad_masks.ndim != expected_ndim:
        msg = f"pad_masks must be 2D, got {pad_masks.ndim}D"
        raise ValueError(msg)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_4d_attention_mask(
    mask_2d: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    min_value: float = -2.3819763e38,
) -> torch.Tensor:
    """Convert 2D attention mask to 4D format for transformers."""  # noqa: DOC201
    mask_4d = mask_2d[:, None, :, :]
    zero = torch.tensor(0.0, dtype=dtype, device=mask_2d.device)
    neg = torch.tensor(min_value, dtype=dtype, device=mask_2d.device)
    return torch.where(mask_4d, zero, neg)

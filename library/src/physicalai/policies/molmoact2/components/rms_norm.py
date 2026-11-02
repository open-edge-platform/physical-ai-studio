# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared RMS normalization for MolmoAct2 components."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS norm with an optional learnable weight."""

    def __init__(self, size: int, *, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        """Build the optional norm weight."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its last dimension.

        Returns:
            The RMS-normalized tensor, preserving the input dtype.
        """
        out_dtype = x.dtype
        normalized = x.to(torch.float32)
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.eps)  # noqa: PLR6104
        normalized = normalized.to(out_dtype)
        return normalized if self.weight is None else normalized * self.weight

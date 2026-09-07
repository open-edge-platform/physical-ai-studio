# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Shared primitive layers used across multiple modules (msat, memory, etc.)."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Shared by action head (MSAT) and memory module.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension to normalize.
            eps: Epsilon added to the variance for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Normalized tensor with the same shape as ``x``.
        """
        input_dtype = x.dtype
        x_float = x.float()
        x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x_normed).to(input_dtype)


def create_norm_layer(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Create normalization layer by name.

    Args:
        norm_type: ``"none"``, ``"layer_norm"``, or ``"rms_norm"``.
        dim: Dimension for normalization.
        eps: Epsilon value.

    Returns:
        Instantiated normalization module.

    Raises:
        ValueError: If ``norm_type`` is not one of the supported values.
    """
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "layer_norm":
        return nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    if norm_type == "rms_norm":
        return RMSNorm(dim, eps=eps)
    msg = f"Unknown norm_type: {norm_type}"
    raise ValueError(msg)


def create_qk_norm_layers(
    qk_norm: str,
    head_dim: int,
    eps: float = 1e-6,
) -> tuple[nn.Module, nn.Module]:
    """Create Q/K normalization layer pair.

    Args:
        qk_norm: ``"none"``, ``"layer_norm"``, or ``"rms_norm"``.
        head_dim: Per-head dimension.
        eps: Epsilon value.

    Returns:
        A ``(q_norm, k_norm)`` pair of normalization modules.

    Raises:
        ValueError: If ``qk_norm`` is not one of the supported values.
    """
    if qk_norm == "none":
        return nn.Identity(), nn.Identity()
    if qk_norm == "layer_norm":
        return (
            nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps),
            nn.LayerNorm(head_dim, elementwise_affine=False, eps=eps),
        )
    if qk_norm == "rms_norm":
        return RMSNorm(head_dim, eps=eps), RMSNorm(head_dim, eps=eps)
    msg = f"Unknown qk_norm: {qk_norm}"
    raise ValueError(msg)

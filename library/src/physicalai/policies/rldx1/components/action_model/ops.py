# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""MSAT utility ops: RoPE SwiGLUFFN, head utils (extracted from msat.py)."""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies in complex form (Llama style).

    Args:
        dim: Head dimension (must be even).
        end: Maximum sequence length.
        theta: Base frequency parameter.

    Returns:
        Complex frequencies of shape (end, dim // 2) as complex64.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis for broadcasting with x (Llama style, adapted for our use case).

    Args:
        freqs_cis: Complex frequencies of shape (B, N, D//2) as complex64.
        x: Tensor to broadcast with, of shape (B, H, N, D//2) as complex.

    Returns:
        Reshaped freqs_cis of shape (B, 1, N, D//2) for broadcasting.

    Raises:
        ValueError: If ``x`` does not have 4 dims, ``freqs_cis`` does not have
            3 dims, or their batch/sequence/half-head dimensions mismatch.
    """
    ndim = x.ndim
    if ndim != 4:  # noqa: PLR2004
        msg = f"x should have 4 dims (B, H, N, D//2), got {ndim}"
        raise ValueError(msg)
    if freqs_cis.ndim != 3:  # noqa: PLR2004
        msg = f"freqs_cis should have 3 dims (B, N, D//2), got {freqs_cis.ndim}"
        raise ValueError(msg)
    expected = (x.shape[0], x.shape[2], x.shape[-1])
    if freqs_cis.shape != expected:
        msg = f"freqs_cis shape {freqs_cis.shape} != (B={x.shape[0]}, N={x.shape[2]}, D//2={x.shape[-1]})"
        raise ValueError(msg)

    # Reshape to (B, 1, N, D//2) for broadcasting with (B, H, N, D//2)
    return freqs_cis.unsqueeze(1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Q and K tensors using complex multiplication (Llama style).

    Args:
        xq: Query tensor of shape (B, H, N, D) where D = head_dim.
        xk: Key tensor of shape (B, H, N, D) where D = head_dim.
        freqs_cis: Complex frequencies of shape (B, N, D//2) as complex64.

    Returns:
        Tuple of rotated query and key tensors, each with the same shape as
        the respective input.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RoPEEmbedder1D(nn.Module):
    """Generate RoPE embeddings for 1D sequences with multiple axes (Llama style)."""

    def __init__(
        self,
        head_dim: int,
        axes_dim: list[int],
        theta: float = 10000.0,
        max_seq_len: int = 2048,
    ) -> None:
        """Initialize RoPEEmbedder1D.

        Args:
            head_dim: Total head dimension; must equal ``sum(axes_dim)``.
            axes_dim: Per-axis embedding dimensions.
            theta: Base frequency parameter for RoPE.
            max_seq_len: Maximum sequence length to precompute frequencies for.

        Raises:
            ValueError: If ``sum(axes_dim) != head_dim``.
        """
        super().__init__()
        if sum(axes_dim) != head_dim:
            msg = f"sum(axes_dim)={sum(axes_dim)} must equal head_dim={head_dim}"
            raise ValueError(msg)
        self.head_dim = head_dim
        self.axes_dim = axes_dim
        self.theta = theta
        self.n_axes = len(axes_dim)
        self.max_seq_len = max_seq_len

        for i, axis_dim in enumerate(axes_dim):
            freqs_cis = precompute_freqs_cis(axis_dim, max_seq_len, theta)
            self.register_buffer(f"freqs_cis_{i}", freqs_cis, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute concatenated RoPE frequencies for multi-axis position IDs.

        Args:
            ids: Position IDs of shape (B, N, n_axes).

        Returns:
            Concatenated complex frequencies of shape (B, N, head_dim // 2).

        Raises:
            ValueError: If ``ids.shape[-1] != n_axes``.
        """
        n_axes = ids.shape[-1]
        if n_axes != self.n_axes:
            msg = f"ids.shape[-1]={n_axes} must equal n_axes={self.n_axes}"
            raise ValueError(msg)

        freqs_list = []
        for i in range(n_axes):
            freqs_cis = getattr(self, f"freqs_cis_{i}")
            pos_ids = ids[..., i]
            freqs = freqs_cis[pos_ids]
            freqs_list.append(freqs)

        return torch.cat(freqs_list, dim=-1)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block from LightningDiT."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,  # noqa: ARG002
        drop: float = 0.0,  # noqa: ARG002
        bias: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize SwiGLUFFN.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden dimension; defaults to ``in_features``.
            out_features: Output dimension; defaults to ``in_features``.
            act_layer: Unused; reserved for API compatibility.
            drop: Unused dropout rate; reserved for API compatibility.
            bias: Whether to include bias in linear layers.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feed-forward transformation.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


# RMSNorm, create_norm_layer, create_qk_norm_layers → imported from common.py


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape (B, N, D) to (B, H, N, D_head) for multi-head attention.

    Args:
        x: Input tensor of shape (B, N, D).
        num_heads: Number of attention heads H; D must be divisible by H.

    Returns:
        Contiguous tensor of shape (B, H, N, D // H).
    """
    b, n, d = x.shape
    d_head = d // num_heads
    return x.view(b, n, num_heads, d_head).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape (B, H, N, D_head) to (B, N, D) after multi-head attention.

    Args:
        x: Input tensor of shape (B, H, N, D_head).

    Returns:
        Contiguous tensor of shape (B, N, H * D_head).
    """
    b, h, n, d_head = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(b, n, h * d_head)

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""MSAT utility ops: RoPE SwiGLUFFN, head utils (extracted from msat.py)."""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE cos/sin tables (Llama style, real-valued for export).

    Real-valued equivalent of the complex ``polar`` form: OpenVINO/ONNX cannot
    faithfully lower ``view_as_complex`` / complex multiply, so the rotation is
    carried as a real ``[cos, sin]`` pair instead.

    Args:
        dim: Head dimension (must be even).
        end: Maximum sequence length.
        theta: Base frequency parameter.

    Returns:
        Real tensor of shape ``(end, dim // 2, 2)`` with ``[..., 0] = cos`` and
        ``[..., 1] = sin``.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # (end, dim // 2)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (end, dim // 2, 2)


def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(B, N, D//2)`` cos/sin table to ``(B, 1, N, D//2)`` for broadcast.

    Args:
        freqs: Real cos or sin table of shape ``(B, N, D//2)``.
        x: Query/key tensor of shape ``(B, H, N, D)`` (full head dim).

    Returns:
        ``freqs`` reshaped to ``(B, 1, N, D//2)``.

    Raises:
        ValueError: If ``x`` does not have 4 dims, ``freqs`` does not have 3
            dims, or their batch/sequence/half-head dimensions mismatch.
    """
    ndim = x.ndim
    if ndim != 4:  # noqa: PLR2004
        msg = f"x should have 4 dims (B, H, N, D), got {ndim}"
        raise ValueError(msg)
    if freqs.ndim != 3:  # noqa: PLR2004
        msg = f"freqs should have 3 dims (B, N, D//2), got {freqs.ndim}"
        raise ValueError(msg)
    expected = (x.shape[0], x.shape[2], x.shape[-1] // 2)
    if freqs.shape != expected:
        msg = f"freqs shape {freqs.shape} != (B={x.shape[0]}, N={x.shape[2]}, D//2={x.shape[-1] // 2})"
        raise ValueError(msg)

    return freqs.unsqueeze(1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Q and K using real arithmetic (Llama style).

    Mathematically identical to the complex ``view_as_complex`` multiply, but
    kept real so the exported OpenVINO/ONNX graph is faithful. Adjacent feature
    pairs ``(x[2k], x[2k+1])`` are rotated by ``(cos_k, sin_k)`` and re-interleaved.

    Args:
        xq: Query tensor of shape (B, H, N, D) where D = head_dim.
        xk: Key tensor of shape (B, H, N, D) where D = head_dim.
        freqs_cis: Real cos/sin table of shape ``(B, N, D//2, 2)``.

    Returns:
        Tuple of rotated query and key tensors, each with the same shape as
        the respective input.
    """
    cos = reshape_for_broadcast(freqs_cis[..., 0], xq)  # (B, 1, N, D//2)
    sin = reshape_for_broadcast(freqs_cis[..., 1], xq)

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)  # (B, H, N, D//2, 2)
        x_r, x_i = x_[..., 0], x_[..., 1]
        out_r = x_r * cos - x_i * sin
        out_i = x_r * sin + x_i * cos
        return torch.stack([out_r, out_i], dim=-1).flatten(3)  # (B, H, N, D)

    return _rotate(xq).type_as(xq), _rotate(xk).type_as(xk)


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
            Concatenated real cos/sin table of shape ``(B, N, head_dim // 2, 2)``.

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

        # Concatenate on the frequency axis (-2), keeping the trailing [cos, sin] pair.
        return torch.cat(freqs_list, dim=-2)


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

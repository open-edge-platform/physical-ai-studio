# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native checkpoint-compatible action-conditioned V-JEPA predictor.

Based on ginwind/VLA-JEPA's MIT-licensed vj2_predictor and vj2_modules.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


def _rotate(x: Tensor, positions: Tensor) -> Tensor:
    width = x.shape[-1]
    frequencies = 1.0 / 10000 ** (torch.arange(width // 2, device=x.device, dtype=x.dtype) / (width / 2))
    angles = positions[:, None].to(x) * frequencies
    sine, cosine = angles.sin().repeat(1, 2), angles.cos().repeat(1, 2)
    pairs = x.unflatten(-1, (-1, 2))
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return x * cosine + rotated * sine


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        """Initialize the released predictor's QKV and output projections."""
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.rotary_dim = 2 * ((self.head_dim // 3) // 2)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _rope(self, x: Tensor, positions: Tensor) -> Tensor:
        width = self.rotary_dim
        rotated = [_rotate(x[..., axis * width : (axis + 1) * width], positions[axis]) for axis in range(3)]
        return torch.cat([*rotated, x[..., 3 * width :]], dim=-1)

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        """Return spatial/temporal rotary attention with a frame-causal mask."""
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        output = functional.scaled_dot_product_attention(
            self._rope(query, positions),
            self._rope(key, positions),
            value,
            attn_mask=mask,
        )
        return self.proj(output.transpose(1, 2).reshape(batch, length, dim))


class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        """Initialize the released GELU feed-forward network."""
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x: Tensor) -> Tensor:
        """Return the feed-forward residual."""
        return self.fc2(functional.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float) -> None:
        """Initialize one predictor transformer block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _MLP(dim, ratio)

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        """Return attention and feed-forward residual updates."""
        x = self.attn(self.norm1(x), positions, mask) + x
        return self.mlp(self.norm2(x)) + x


class VideoPredictor(nn.Module):
    """Predict next-step video features from context features and learned latent actions."""

    def __init__(
        self,
        *,
        encoder_dim: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        context_frames: int,
        grid_size: int,
        action_tokens_per_frame: int,
    ) -> None:
        """Initialize the native predictor without changing released parameter names."""
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_dim, hidden_size)
        self.action_encoder = nn.Linear(action_dim, hidden_size)
        # These unused projections are retained for strict checkpoint compatibility.
        self.state_encoder = nn.Linear(action_dim, hidden_size).requires_grad_(requires_grad=False)
        self.extrinsics_encoder = nn.Linear(action_dim - 1, hidden_size).requires_grad_(requires_grad=False)
        self.predictor_blocks = nn.ModuleList([_Block(hidden_size, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.predictor_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.predictor_proj = nn.Linear(hidden_size, encoder_dim)
        self.context_frames = context_frames
        self.patches_per_frame = grid_size * grid_size
        self.action_tokens_per_frame = action_tokens_per_frame
        self.gradient_checkpointing = False
        tokens_per_frame = action_tokens_per_frame + self.patches_per_frame
        times = torch.arange(context_frames).repeat_interleave(tokens_per_frame)
        spatial = torch.arange(self.patches_per_frame)
        rows = torch.cat((torch.zeros(action_tokens_per_frame), spatial // grid_size)).repeat(context_frames)
        columns = torch.cat((torch.zeros(action_tokens_per_frame), spatial % grid_size)).repeat(context_frames)
        self.register_buffer("positions", torch.stack((times, rows, columns)), persistent=False)
        self.register_buffer("attention_mask", times[:, None] >= times[None, :], persistent=False)

    def forward(self, frame_tokens: Tensor, action_tokens: Tensor) -> Tensor:
        """Return future video features for each context frame and spatial patch."""
        batch = frame_tokens.shape[0]
        frames = self.predictor_embed(frame_tokens).view(batch, self.context_frames, self.patches_per_frame, -1)
        actions = self.action_encoder(action_tokens).view(batch, self.context_frames, self.action_tokens_per_frame, -1)
        x = torch.cat((actions, frames), dim=2).flatten(1, 2)
        for block in self.predictor_blocks:
            if self.gradient_checkpointing and self.training:
                x = cast("Tensor", checkpoint(block, x, self.positions, self.attention_mask, use_reentrant=False))
            else:
                x = block(x, self.positions, self.attention_mask)
        x = x.view(batch, self.context_frames, self.action_tokens_per_frame + self.patches_per_frame, -1)
        x = x[:, :, self.action_tokens_per_frame :].flatten(1, 2)
        return self.predictor_proj(self.predictor_norm(x))

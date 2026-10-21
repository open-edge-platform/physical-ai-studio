# Copyright 2025 NVIDIA Corp. and affiliates.
# Copyright 2026 The HuggingFace Inc. team.
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-compatible VLA-JEPA flow-matching DiT action head."""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import torch
from diffusers.models.attention import Attention, FeedForward
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.checkpoint import checkpoint

from physicalai.policies.components import SinusoidalPositionalEncoding
from physicalai.policies.components.action_heads import IterativeActionHead
from physicalai.policies.components.nn import TimestepEncoder


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    layers: OrderedDict[str, nn.Module] = OrderedDict([
        ("layer1", nn.Linear(input_dim, hidden_dim)),
        ("relu", nn.ReLU()),
        ("layer2", nn.Linear(hidden_dim, output_dim)),
    ])
    return nn.Sequential(layers)


class ActionEncoder(nn.Module):
    """Embed noisy actions and discretized flow time."""

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        """Initialize action and timestep projections."""
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        """Return action/time embeddings."""
        actions = self.layer1(actions)
        times = self.pos_encoding(timesteps[:, None].expand(-1, actions.shape[1])).to(actions.dtype)
        return self.layer3(F.silu(self.layer2(torch.cat((actions, times), dim=-1))))


class AdaLayerNorm(nn.Module):
    """Time-conditioned layer normalization with the released parameter names."""

    def __init__(self, dim: int) -> None:
        """Initialize learned scale and shift projections."""
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """Return the normalized and modulated token sequence."""
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class TransformerBlock(nn.Module):
    """Alternate cross-attention and self-attention, as in the pretrained DiT."""

    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float, context_dim: int | None) -> None:
        """Initialize one alternating-attention block."""
        super().__init__()
        self.is_cross_attention = context_dim is not None
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=True,
            cross_attention_dim=context_dim,
            out_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="gelu-approximate", final_dropout=True)

    def forward(self, x: Tensor, tokens: Tensor, temb: Tensor) -> Tensor:
        """Return attention and feed-forward residual updates."""
        context = tokens if self.is_cross_attention else None
        x = self.attn1(self.norm1(x, temb), encoder_hidden_states=context) + x
        return x + self.ff(self.norm2(x))


class DiT(nn.Module):
    """Diffusion transformer preserving the upstream state-dict layout."""

    def __init__(
        self,
        heads: int,
        head_dim: int,
        output_dim: int,
        layers: int,
        dropout: float,
        context_dim: int,
    ) -> None:
        """Initialize the checkpoint-compatible DiT stack."""
        super().__init__()
        dim = heads * head_dim
        self.timestep_encoder = TimestepEncoder(dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, head_dim, dropout, context_dim if index % 2 == 0 else None)
            for index in range(layers)
        ])
        self.norm_out = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(dim, dim * 2)
        self.proj_out_2 = nn.Linear(dim, output_dim)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor, tokens: Tensor, timesteps: Tensor) -> Tensor:
        """Return time-conditioned hidden states."""
        temb = self.timestep_encoder(timesteps)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training:
                x = cast("Tensor", checkpoint(block, x, tokens, temb, use_reentrant=False))
            else:
                x = block(x, tokens, temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        return self.proj_out_2(self.norm_out(x) * (1 + scale[:, None]) + shift[:, None])


class VLAJEPAActionHead(IterativeActionHead):
    """Generate action chunks with the released VLA-JEPA flow convention."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        context_dim: int,
        action_dim: int,
        state_dim: int,
        chunk_size: int,
        hidden_size: int = 1024,
        num_layers: int = 16,
        num_heads: int = 12,
        head_dim: int = 64,
        dropout: float = 0.2,
        num_future_tokens: int = 32,
        max_seq_len: int = 1024,
        num_inference_steps: int = 4,
        num_timestep_buckets: int = 1000,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
    ) -> None:
        """Initialize robot projections, DiT, and flow settings."""
        super().__init__(chunk_size, action_dim, num_inference_steps)
        dim = num_heads * head_dim
        self.model = DiT(num_heads, head_dim, hidden_size, num_layers, dropout, context_dim)
        self.action_encoder = ActionEncoder(action_dim, dim)
        self.action_decoder = _mlp(hidden_size, hidden_size, action_dim)
        self.state_encoder = _mlp(state_dim, hidden_size, dim) if state_dim else None
        self.future_tokens = nn.Embedding(num_future_tokens, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s

    def denoise(self, x_t: Tensor, t: Tensor, context: dict[str, Tensor]) -> Tensor:
        """Return flow velocity using tensor-only time discretization."""
        times = (t * self.num_timestep_buckets).long()
        actions = self.action_encoder(x_t, times)
        positions = torch.arange(x_t.shape[1], device=x_t.device)
        actions = self.position_embedding(positions)[None] + actions
        future = self.future_tokens.weight[None].expand(x_t.shape[0], -1, -1)
        parts = [future, actions]
        if self.state_encoder is not None and "state" in context:
            parts.insert(0, self.state_encoder(context["state"]))
        hidden = self.model(torch.cat(parts, dim=1), context["tokens"], times)
        return self.action_decoder(hidden[:, -x_t.shape[1] :])

    def compute_loss(self, actions: Tensor, context: dict[str, Tensor]) -> Tensor:
        """Return unreduced per-element flow loss."""
        noise = torch.randn_like(actions)
        beta = torch.distributions.Beta(self.noise_beta_alpha, self.noise_beta_beta)
        times = (self.noise_s - beta.sample((actions.shape[0],)).to(actions)) / self.noise_s
        noisy = (1 - times[:, None, None]) * noise + times[:, None, None] * actions
        return (self.denoise(noisy, times, context) - (actions - noise)).square()

    @staticmethod
    def timesteps(num_steps: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the noise-to-data Euler schedule."""
        return torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    @staticmethod
    def step(x_t: Tensor, model_output: Tensor, t: Tensor, t_next: Tensor) -> Tensor:
        """Return one Euler update."""
        return x_t + (t_next - t) * model_output

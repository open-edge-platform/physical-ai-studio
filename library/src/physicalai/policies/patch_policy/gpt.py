# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2022 Andrej Karpathy
# SPDX-License-Identifier: MIT

"""Patch-aware causal transformer trunk for Patch Policy.

This is an adapted nanoGPT implementation: it keeps the original Karpathy block-causal
attention logic, while reshaping patch tokens from ``[B, T, P, E]`` into the flattened
sequence expected by the transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TOKEN_NDIM = 4


def new_gelu(x: Tensor) -> Tensor:
    """Karpathy's GPT GELU implementation."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def generate_mask_matrix(npatch: int, nwindow: int) -> Tensor:
    """Build the patch-level causal mask used by block-causal attention."""
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows: list[Tensor] = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention under a block-causal patch mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})."
            raise ValueError(msg)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", generate_mask_matrix(config.n_patches, config.block_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.bias[:, :, :t, :t]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise MLP block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Single GPT transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """Config for the patch-aware GPT trunk."""

    block_size: int = 1024
    input_dim: int = 256
    output_dim: int = 256
    n_patches: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class PatchGPT(nn.Module):
    """Causal transformer over per-timestep patch tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_patches: int,
        n_obs_steps: int,
        n_layer: int = 8,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = GPTConfig(
            block_size=n_obs_steps,
            input_dim=input_dim,
            output_dim=output_dim,
            n_patches=n_patches,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
        )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(self.config.input_dim, self.config.n_embd),
                wpe=nn.Embedding(self.config.block_size * self.config.n_patches, self.config.n_embd),
                drop=nn.Dropout(self.config.dropout),
                h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                ln_f=nn.LayerNorm(self.config.n_embd),
            ),
        )
        self.lm_head = nn.Linear(self.config.n_embd, self.config.output_dim, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != TOKEN_NDIM:
            msg = f"Expected tokens with shape [B, T, P, E], got {tuple(tokens.shape)}."
            raise ValueError(msg)

        b, t, p, _ = tokens.size()
        if t > self.config.block_size:
            msg = f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}."
            raise ValueError(msg)
        if p != self.config.n_patches:
            msg = f"Expected {self.config.n_patches} patches per timestep, got {p}."
            raise ValueError(msg)

        pos = torch.arange(0, t * p, dtype=torch.long, device=tokens.device).unsqueeze(0)
        tok_emb = self.transformer.wte(tokens)
        pos_emb = self.transformer.wpe(pos).reshape(1, t, p, self.config.n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        x = x.reshape(b, t * p, self.config.n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = logits.reshape(b, t, p, self.config.output_dim)
        return logits[:, :, -1, :]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def crop_block_size(self, block_size: int) -> None:
        """Trim the learned position embeddings to a smaller observation window."""
        if block_size > self.config.block_size:
            msg = f"Cannot crop to a larger block size: {block_size} > {self.config.block_size}."
            raise ValueError(msg)
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[: block_size * self.config.n_patches])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[
                :, :, : block_size * self.config.n_patches, : block_size * self.config.n_patches
            ]

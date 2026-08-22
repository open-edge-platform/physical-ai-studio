# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Derived from https://github.com/OpenHelix-Team/VLA-Adapter
# (prismatic/models/action_heads.py), Copyright (c) OpenHelix Team,
# licensed under the MIT License.

"""The VLA-Adapter "Policy" head — an MLP-ResNet driven by Bridge Attention.

The paper's contribution. The head never sees the raw observation: its input is
*all zeros*, and every bit of information arrives by cross-attention onto the
frozen VLM's per-layer hidden states.

Block ``i`` is conditioned on LLM layer ``i + 1`` — hence block count must equal
LLM depth. Each layer's states split into the leading ``num_task_tokens``
("task", ``h_t``) and the trailing action-query positions ("action", ``h_a``).
With the projected proprio token ``p`` these form two cross-attention pathways
whose relative strength is a learned, ``tanh``-squashed ``gating_factor`` — the
"Bridge".

Differences from upstream, all behaviour-preserving: shape constants are
constructor args rather than ``sys.argv`` globals; ``phase="Training"`` becomes
``nn.Module.training``; the hard-coded bfloat16 cast is dropped so the head runs
on CPU and through export. Module *structure* is unchanged, so ``state_dict()``
stays compatible with ``action_head--checkpoint.pt``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Standard deviation of the training-time input perturbation. Upstream builds a
# fresh (unregistered, hence untrained) ``nn.Parameter`` on every forward pass,
# which is equivalent to adding fixed-scale Gaussian noise. Keeping it
# unregistered also keeps our ``state_dict`` compatible with the released
# checkpoints, which contain no perturbation entry.
_PERTURBATION_STD = 0.02


class MLPResNetBlock(nn.Module):
    """One residual block with fused self / task / adapter attention.

    One set of Q/K/V projections serves three pathways whose scores are
    concatenated *before* a single softmax, so they compete for one budget of
    attention mass rather than being summed after:

    1. **self** — the block's own token features,
    2. **task** — ``cat(h_a, p)``, action-query and proprio features,
    3. **adapter** — ``h_t``, the vision features, scaled by ``tanh(g)``.

    Note:
        The pathway names are upstream's and read inverted: ``task`` attends to
        the *action* states, ``adapter`` to ``h_t``. Behaviour is preserved
        verbatim; only the ``h_t`` pathway is gated.
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """Initialize the block.

        Args:
            dim: Hidden width. Must divide evenly by ``num_heads``, since
                attention reshapes it into ``num_heads`` subspaces via ``view``.
            num_heads: Attention heads, shared by all three pathways.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
        """
        super().__init__()
        if dim % num_heads != 0:
            msg = f"Hidden width {dim} must be divisible by num_heads {num_heads}."
            raise ValueError(msg)

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, S, C)`` into ``(B, num_heads, S, head_dim)``.

        Args:
            tensor: Input ``(B, S, C)``.

        Returns:
            ``(B, num_heads, S, head_dim)``.
        """
        batch, seq, _ = tensor.shape
        return tensor.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(  # noqa: PLR0914 - the fused three-pathway attention needs them
        self,
        x: torch.Tensor,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one Bridge-Attention residual block.

        Args:
            x: Token features ``(B, T, dim)``.
            h_t: Task (vision) states ``(B, K_t, dim)``.
            h_a: Action states ``(B, K_a, dim)``.
            p: Optional proprio features ``(B, 1, dim)``.

        Returns:
            ``(B, T, dim)``.
        """
        ratio_g = torch.tanh(self.gating_factor)

        conditions = [h_a] if p is None else [h_a, p]
        cond = torch.cat(conditions, dim=1)

        batch, seq, width = x.shape

        query = self._split_heads(self.q_proj(x))
        k_tokens = self._split_heads(self.k_proj(x))
        v_tokens = self._split_heads(self.v_proj(x))
        k_task = self._split_heads(self.k_proj(cond))
        v_task = self._split_heads(self.v_proj(cond))
        k_adapter = self._split_heads(self.k_proj(h_t))
        v_adapter = self._split_heads(self.v_proj(h_t))

        scores_tokens = torch.matmul(query, k_tokens.transpose(-2, -1))
        scores_task = torch.matmul(query, k_task.transpose(-2, -1))
        scores_adapter = torch.matmul(query, k_adapter.transpose(-2, -1)) * ratio_g

        scores = torch.cat([scores_tokens, scores_task, scores_adapter], dim=-1)
        scores /= math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        values = torch.cat([v_tokens, v_task, v_adapter], dim=2)
        attended = torch.matmul(weights, values)

        attended = attended.transpose(1, 2).contiguous().view(batch, seq, width)
        attended = self.o_proj(attended)

        return self.ffn(attended + x)


class MLPResNet(nn.Module):
    """Stack of :class:`MLPResNetBlock` with input and output projections."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 8,
    ) -> None:
        """Initialize the residual stack.

        Args:
            num_blocks: Residual blocks; must match LLM depth, since block ``i``
                is conditioned on layer ``i + 1``.
            input_dim: Flattened input width (``action_dim * hidden_dim``).
            hidden_dim: Working width of the blocks.
            output_dim: Per-step action dimension.
            num_heads: Attention heads per block.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            MLPResNetBlock(dim=hidden_dim, num_heads=num_heads) for _ in range(num_blocks)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_t: torch.Tensor,
        h_a: torch.Tensor,
        p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the stack.

        Args:
            x: Input features ``(B, chunk, input_dim)``.
            h_t: Per-layer task states ``(B, L + 1, K_t, hidden_dim)``.
            h_a: Per-layer action states ``(B, L + 1, K_a, hidden_dim)``.
            p: Optional proprio features ``(B, 1, hidden_dim)``.

        Returns:
            ``(B, chunk, output_dim)``.
        """
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        # Block i reads layer i+1 of the backbone: index 0 is the embedding
        # output, so the first transformer layer sits at index 1.
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t=h_t[:, i + 1], h_a=h_a[:, i + 1], p=p)
        x = self.layer_norm2(x)
        return self.fc2(x)


class L1RegressionActionHead(nn.Module):
    """Continuous action head trained with an L1 objective.

    Produces the whole chunk in a **single forward pass** — no diffusion or
    flow-matching loop — which keeps the exported graph static.
    """

    def __init__(
        self,
        input_dim: int = 896,
        hidden_dim: int = 896,
        action_dim: int = 7,
        chunk_size: int = 8,
        num_task_tokens: int = 512,
        num_blocks: int = 24,
        num_heads: int = 8,
    ) -> None:
        """Initialize the action head.

        Args:
            input_dim: Backbone (LLM) hidden width.
            hidden_dim: Working width of the residual blocks.
            action_dim: Per-step action dimension.
            chunk_size: Action steps produced per call.
            num_task_tokens: Leading positions treated as task features; the
                rest are action-query features.
            num_blocks: Residual blocks; must match the LLM depth.
            num_heads: Attention heads per block.
        """
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        proprio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an action chunk from stacked per-layer hidden states.

        Args:
            hidden_states: Layer-stacked states ``(B, L + 1, S, hidden_dim)``.
            proprio_features: Optional proprio token ``(B, 1, hidden_dim)``.

        Returns:
            ``(B, chunk_size, action_dim)``.
        """
        batch = hidden_states.shape[0]

        task_states = hidden_states[:, :, : self.num_task_tokens, :]
        action_states = hidden_states[:, :, self.num_task_tokens :, :]

        # The head's own input carries no information: it is a zero tensor of
        # shape (B, chunk, action_dim * hidden_dim). Everything the head knows
        # arrives through the cross-attention pathways below.
        x = torch.zeros(
            (batch, self.chunk_size, self.action_dim * self.hidden_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if self.training:
            x += torch.randn_like(x) * _PERTURBATION_STD

        return self.model(x, h_t=task_states, h_a=action_states, p=proprio_features)

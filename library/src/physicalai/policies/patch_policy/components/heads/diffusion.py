# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Upstream-compatible patch-conditioned diffusion action head."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from physicalai.policies.components import DiffusionActionHead

if TYPE_CHECKING:
    from physicalai.policies.components.action_heads import Context
    from physicalai.policies.components.action_heads.diffusion.diffusion import BetaSchedule, PredictionType


class _SinusoidalPositionalEmbedding(nn.Module):
    """Timestep embedding used by the original Patch Policy."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        """Encode one timestep per batch item.

        Returns:
            Sinusoidal timestep embeddings for the batch.
        """
        half_dim = self.hidden_dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
        values = timesteps[:, None] * frequencies[None]
        return torch.cat((values.sin(), values.cos()), dim=-1)


class DiffusionHead(DiffusionActionHead):
    """Patch Policy's causal transformer diffusion network."""

    def __init__(  # noqa: PLR0913
        self,
        context_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        n_obs_steps: int,
        n_patches: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: BetaSchedule = "squaredcos_cap_v2",
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
    ) -> None:
        """Initialize the diffusion scheduler and causal transformer network."""
        self.action_horizon = chunk_size
        self.n_obs_steps = n_obs_steps
        self.n_patches = n_patches
        prediction_horizon = n_obs_steps + chunk_size - 1
        super().__init__(
            prediction_horizon,
            action_dim,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            eta=eta,
            use_random_input_noise=use_random_input_noise,
        )
        self.action_in = nn.Linear(action_dim, hidden_dim)
        self.context_in = nn.Linear(context_dim, hidden_dim)
        self.time_encoder = _SinusoidalPositionalEmbedding(hidden_dim)
        self.action_position = nn.Parameter(torch.zeros(1, prediction_horizon, hidden_dim))
        context_length = 1 + n_obs_steps * n_patches
        self.context_position = nn.Parameter(torch.zeros(1, context_length, hidden_dim))
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.Mish(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.input_dropout = nn.Dropout(embedding_dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("target_mask", self._target_mask(prediction_horizon), persistent=False)
        self.register_buffer(
            "memory_mask",
            self._memory_mask(prediction_horizon, n_obs_steps, n_patches),
            persistent=False,
        )
        self.apply(self._initialize)
        nn.init.normal_(self.action_position, mean=0.0, std=0.02)
        nn.init.normal_(self.context_position, mean=0.0, std=0.02)

    @staticmethod
    def _target_mask(length: int) -> Tensor:
        allowed = torch.ones(length, length, dtype=torch.bool).tril()
        return torch.zeros(length, length).masked_fill(~allowed, float("-inf"))

    @staticmethod
    def _memory_mask(action_steps: int, observation_steps: int, patches: int) -> Tensor:
        allowed = torch.zeros(action_steps, 1 + observation_steps * patches, dtype=torch.bool)
        allowed[:, 0] = True
        for action_step in range(action_steps):
            observed = min(action_step, observation_steps - 1) + 1
            allowed[action_step, 1 : 1 + observed * patches] = True
        return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight" in name:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def denoise(self, x_t: Tensor, t: Tensor, context: Context) -> Tensor:
        """Predict noise from actions, timestep, and causal visual context.

        Returns:
            Predicted diffusion targets with the noisy action tensor's shape.
        """
        tokens = context["tokens"]
        time = self.time_encoder(t.to(x_t.dtype))[:, None]
        memory = torch.cat((time, self.context_in(tokens)), dim=1)
        memory = self.context_encoder(self.input_dropout(memory + self.context_position))
        hidden = self.input_dropout(self.action_in(x_t) + self.action_position)
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=self.target_mask,
            memory_mask=self.memory_mask,
        )
        return self.action_out(self.norm(decoded))

    def sample(self, context: Context, *, noise: Tensor | None = None, num_steps: int | None = None) -> Tensor:
        """Sample the full trajectory and return the current action chunk.

        Returns:
            Predicted actions shaped ``[B, action_horizon, action_dim]``.
        """
        trajectory = super().sample(context, noise=noise, num_steps=num_steps)
        start = self.n_obs_steps - 1
        return trajectory[:, start : start + self.action_horizon]

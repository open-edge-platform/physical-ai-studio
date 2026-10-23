# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch-conditioned diffusion action head."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from physicalai.policies.components import DiffusionActionHead, SinusoidalPositionalEncoding

if TYPE_CHECKING:
    from physicalai.policies.components.action_heads import Context
    from physicalai.policies.components.action_heads.diffusion.diffusion import BetaSchedule, PredictionType


class PatchDiffusionHead(DiffusionActionHead):
    """Denoise action tokens by cross-attending to visual patch tokens.

    Args:
        context_dim: Width of each visual context token.
        action_dim: Width of each action timestep.
        chunk_size: Number of action timesteps predicted per sample.
        hidden_dim: Transformer decoder hidden width.
        num_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads in each decoder layer.
        dropout: Dropout probability used by decoder layers.
        num_train_timesteps: Number of diffusion timesteps used for training.
        num_inference_steps: Number of denoising steps used for sampling.
        beta_schedule: Noise schedule used by the diffusion scheduler.
        prediction_type: Diffusion target represented by the denoiser output.
        clip_sample: Whether to clip samples after each scheduler step.
        clip_sample_range: Symmetric range used to clip samples.
        eta: DDIM stochasticity coefficient used during sampling.
        use_random_input_noise: Whether sampling starts from random noise.
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_schedule: BetaSchedule = "squaredcos_cap_v2",
        prediction_type: PredictionType = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        eta: float = 1.0,
        use_random_input_noise: bool = True,
    ) -> None:
        super().__init__(
            chunk_size,
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
        self.time_encoder = SinusoidalPositionalEncoding(hidden_dim)
        self.action_position = nn.Parameter(torch.zeros(1, chunk_size, hidden_dim))
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

    def prepare_context(self, context: Context) -> Context:
        """Project visual tokens once for the complete denoising loop.

        Args:
            context: Context containing visual tokens or preprojected memory.

        Returns:
            Context containing projected visual memory.
        """
        if "memory" in context:
            return context
        return {**context, "memory": self.context_in(context["tokens"])}

    def denoise(self, x_t: Tensor, t: Tensor, context: Context) -> Tensor:
        """Predict noise or clean actions from noisy actions and visual memory.

        Args:
            x_t: Noisy action chunks at the current diffusion timestep.
            t: Diffusion timestep for each batch item.
            context: Context containing visual tokens or projected memory.

        Returns:
            The configured diffusion prediction for each action element.
        """
        time = self.time_encoder(t[:, None].to(x_t.dtype))
        hidden = self.action_in(x_t) + self.action_position + time
        memory = context.get("memory")
        if memory is None:
            memory = self.context_in(context["tokens"])
        return self.action_out(self.norm(self.decoder(hidden, memory)))

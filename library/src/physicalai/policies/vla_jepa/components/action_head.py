# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: FBT001, FBT002, PLR6104
# ^^^^ Disabled because this module follows the diffusers API conventions the published
# checkpoints were trained against, which use boolean positional arguments.

"""Flow-matching DiT action head for VLA-JEPA.

Ported from LeRobot's ``lerobot.policies.vla_jepa.action_head``. Class and attribute names
are kept identical to the LeRobot implementation because they determine the published
checkpoint keys.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.policies.vla_jepa.config import VLAJEPAConfig

try:
    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
    from diffusers.models.attention import Attention, FeedForward
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps

    _DIFFUSERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional extra

    class ModelMixin:  # type: ignore[no-redef]
        """Stub base used when diffusers is not installed."""

    class ConfigMixin:  # type: ignore[no-redef]
        """Stub base used when diffusers is not installed."""

    def register_to_config(func: Callable) -> Callable:  # type: ignore[no-redef]
        """Return the decorated function unchanged when diffusers is missing.

        Args:
            func: The ``__init__`` that diffusers would otherwise wrap.

        Returns:
            The function unchanged.
        """
        return func

    Attention = FeedForward = TimestepEmbedding = Timesteps = None
    _DIFFUSERS_AVAILABLE = False


def _require_diffusers() -> None:
    """Fail with an actionable message when the diffusers extra is missing.

    Raises:
        ImportError: If diffusers is not installed.
    """
    if not _DIFFUSERS_AVAILABLE:
        msg = (
            "VLA-JEPA's action head requires diffusers.\n\nInstall with:\n"
            "    uv pip install 'physicalai-train[vla_jepa]'"
        )
        raise ImportError(msg)


class SinusoidalPositionalEncoding(nn.Module):
    """Sine/cosine encoding of the flow-matching timestep, broadcast over the chunk."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize the encoding.

        Args:
            embedding_dim: Width of the produced encoding.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode timesteps.

        Args:
            timesteps: Timesteps of shape ``[B, T]``.

        Returns:
            Encoding of shape ``[B, T, embedding_dim]``.
        """
        timesteps = timesteps.float()
        batch_size, seq_len = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim, 1))
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).view(batch_size, seq_len, -1)


class ActionEncoder(nn.Module):
    """Embeds noisy actions together with their flow-matching timestep."""

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        """Initialize the action encoder.

        Args:
            action_dim: Action dimensionality.
            hidden_size: Width of the produced tokens.
        """
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed an action chunk at a given timestep.

        Args:
            actions: Actions of shape ``[B, T, action_dim]``.
            timesteps: Timesteps of shape ``[B]``.

        Returns:
            Action tokens of shape ``[B, T, hidden_size]``.

        Raises:
            ValueError: If `timesteps` is not one value per batch item.
        """
        batch_size, seq_len, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            msg = "timesteps must have shape [batch_size]."
            raise ValueError(msg)
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        return self.layer3(F.silu(self.layer2(torch.cat([action_emb, time_emb], dim=-1))))


class TimestepEncoder(nn.Module):
    """Maps the diffusion timestep to the DiT's conditioning embedding."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize the timestep encoder.

        Args:
            embedding_dim: Width of the produced embedding.
        """
        super().__init__()
        _require_diffusers()
        self.time_proj = Timesteps(  # pyrefly: ignore[not-callable]
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(  # pyrefly: ignore[not-callable]
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed the timestep.

        Args:
            timesteps: Timesteps of shape ``[B]``.

        Returns:
            Embedding of shape ``[B, embedding_dim]``.
        """
        projected = self.time_proj(timesteps).to(dtype=next(self.parameters()).dtype)
        return self.timestep_embedder(projected)


class AdaLayerNorm(nn.Module):
    """Layer norm whose scale and shift are predicted from the timestep embedding."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize the adaptive layer norm.

        Args:
            embedding_dim: Token width.
        """
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Normalize and modulate the tokens.

        Args:
            x: Input tokens.
            temb: Timestep embedding.

        Returns:
            The modulated tokens.
        """
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """One DiT block: adaptive-norm attention plus a feed-forward layer."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        cross_attention_dim: int,
        is_cross_attention: bool = True,
    ) -> None:
        """Initialize the block.

        Args:
            dim: Token width.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Width per attention head.
            dropout: Dropout probability.
            cross_attention_dim: Width of the conditioning tokens.
            is_cross_attention: Whether this block cross-attends to the conditioning tokens.
        """
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(  # pyrefly: ignore[not-callable]
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=True,
            cross_attention_dim=cross_attention_dim,
            out_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(  # pyrefly: ignore[not-callable]
            dim,
            dropout=dropout,
            activation_fn="gelu-approximate",
            final_dropout=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Run attention and the feed-forward layer with residual connections.

        Args:
            hidden_states: The block's own token sequence.
            encoder_hidden_states: Conditioning tokens for cross-attention.
            temb: Timestep embedding.

        Returns:
            The updated tokens.
        """
        attn_input = self.norm1(hidden_states, temb)
        attention_context = encoder_hidden_states if self.is_cross_attention else None
        hidden_states = hidden_states + self.attn1(attn_input, encoder_hidden_states=attention_context)
        return hidden_states + self.ff(self.norm2(hidden_states))


class DiT(ModelMixin, ConfigMixin):  # pyrefly: ignore[invalid-inheritance]
    """Diffusion transformer predicting the flow-matching velocity of an action chunk."""

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cross_attention_dim: int,
    ) -> None:
        """Initialize the transformer.

        Args:
            num_attention_heads: Number of attention heads.
            attention_head_dim: Width per attention head.
            output_dim: Width of the projected output.
            num_layers: Number of transformer blocks.
            dropout: Dropout probability.
            cross_attention_dim: Width of the conditioning tokens.
        """
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim if layer_idx % 2 == 0 else self.inner_dim,
                    is_cross_attention=layer_idx % 2 == 0,
                )
                for layer_idx in range(num_layers)
            ],
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(self.inner_dim, self.inner_dim * 2)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer.

        Args:
            hidden_states: The DiT's own token sequence.
            encoder_hidden_states: Conditioning tokens.
            timestep: Discretized flow-matching timestep.

        Returns:
            The projected output tokens.
        """
        temb = self.timestep_encoder(timestep)
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, temb=temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)


@dataclass
class ActionModelPreset:
    """Default head geometry per `action_model_type`.

    Only the attention geometry is preset; the DiT's width comes from
    `config.action_hidden_size`, so there is deliberately no `hidden_size` here.
    """

    attention_head_dim: int
    num_attention_heads: int


# A state without a time dimension: [B, state_dim] rather than [B, 1, state_dim].
_UNBATCHED_STATE_DIMS = 2

DIT_PRESETS = {
    "DiT-B": ActionModelPreset(attention_head_dim=64, num_attention_heads=12),
    "DiT-L": ActionModelPreset(attention_head_dim=48, num_attention_heads=32),
    "DiT-test": ActionModelPreset(attention_head_dim=8, num_attention_heads=2),
}


class VLAJEPAActionHead(nn.Module):
    """Flow-matching action head conditioned on the backbone's embodied-action tokens."""

    def __init__(self, config: VLAJEPAConfig, cross_attention_dim: int) -> None:
        """Initialize the action head.

        Args:
            config: Policy configuration.
            cross_attention_dim: Width of the backbone's hidden states.
        """
        super().__init__()
        _require_diffusers()
        preset = DIT_PRESETS[config.action_model_type]
        self.config = config
        num_heads = config.action_num_heads or preset.num_attention_heads
        head_dim = config.action_attention_head_dim or preset.attention_head_dim
        inner_dim = num_heads * head_dim  # e.g. DiT-B: 12 heads x 64 = 768

        self.input_embedding_dim = inner_dim
        self.action_horizon = config.chunk_size
        self.num_inference_timesteps = config.num_inference_timesteps

        hidden_size = config.action_hidden_size
        self.model = DiT(
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            output_dim=hidden_size,
            num_layers=config.action_num_layers,
            dropout=config.action_dropout,
            cross_attention_dim=cross_attention_dim,
        )
        self.action_encoder = ActionEncoder(config.action_dim, inner_dim)
        self.action_decoder = nn.Sequential(  # pyrefly: ignore[no-matching-overload]
            OrderedDict(
                [
                    ("layer1", nn.Linear(hidden_size, hidden_size)),
                    ("relu", nn.ReLU()),
                    ("layer2", nn.Linear(hidden_size, config.action_dim)),
                ],
            ),
        )
        self.state_encoder = (
            nn.Sequential(  # pyrefly: ignore[no-matching-overload]
                OrderedDict(
                    [
                        ("layer1", nn.Linear(config.state_dim, hidden_size)),
                        ("relu", nn.ReLU()),
                        ("layer2", nn.Linear(hidden_size, inner_dim)),
                    ],
                ),
            )
            if config.state_dim > 0
            else None
        )
        self.future_tokens = nn.Embedding(config.num_embodied_action_tokens_per_instruction, inner_dim)
        self.position_embedding = nn.Embedding(
            max(
                config.action_max_seq_len,
                config.chunk_size + config.num_action_tokens_per_timestep + 4,
            ),
            inner_dim,
        )
        self.beta_dist = Beta(config.action_noise_beta_alpha, config.action_noise_beta_beta)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample flow-matching timesteps from the configured Beta distribution.

        Args:
            batch_size: Number of samples to draw.
            device: Device to place the samples on.
            dtype: Dtype of the samples.

        Returns:
            Timesteps of shape ``[batch_size]``.
        """
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.action_noise_s - sample) / self.config.action_noise_s

    def _build_inputs(
        self,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Build the DiT's own token sequence: [state?, future queries, noisy actions].

        The conditioning tokens are not part of this sequence; they reach the DiT as
        `encoder_hidden_states` through cross-attention.

        Args:
            actions: Noisy action chunk.
            state: Current state, or None.
            timesteps: Discretized flow-matching timestep.

        Returns:
            The DiT input sequence.
        """
        action_features = self.action_encoder(actions, timesteps)
        pos_ids = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + self.position_embedding(pos_ids)[None]

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        seq = [future_tokens, action_features]
        if state is not None and self.state_encoder is not None:
            if state.ndim == _UNBATCHED_STATE_DIMS:
                state = state.unsqueeze(1)
            seq.insert(0, self.state_encoder(state))
        return torch.cat(seq, dim=1)

    def forward(
        self,
        conditioning_tokens: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute the flow-matching training loss.

        Args:
            conditioning_tokens: Backbone hidden states used as cross-attention context.
            actions: Ground-truth action chunk ``[B, T, action_dim]``.
            state: Current state, or None.
            action_is_pad: Padding mask ``[B, T]``, or None.
            reduction: ``"mean"`` for the scalar loss, ``"none"`` for a per-sample loss.

        Returns:
            The masked velocity-prediction loss.
        """
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy_actions = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity = actions - noise
        t_discretized = (t * self.config.action_num_timestep_buckets).long()

        hidden_states = self._build_inputs(noisy_actions, state, t_discretized)
        pred = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=t_discretized,
        )
        pred_actions = self.action_decoder(pred[:, -actions.shape[1] :])

        if action_is_pad is None:
            action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)

        loss = F.mse_loss(pred_actions, velocity, reduction="none")  # [B, T, action_dim]
        valid_mask = ~action_is_pad.unsqueeze(-1)  # [B, T, 1]
        if reduction == "none":
            # Per-sample loss (B,) for sample weighting (RA-BC): mask-average over T and action_dim.
            per_sample_valid = valid_mask.sum(dim=(1, 2)) * loss.shape[-1]  # [B]
            return (loss * valid_mask).sum(dim=(1, 2)) / per_sample_valid.clamp_min(1)
        num_valid = valid_mask.sum() * loss.shape[-1]
        return (loss * valid_mask).sum() / num_valid.clamp_min(1)

    @torch.no_grad()
    def predict_action(
        self,
        conditioning_tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the flow from noise to an action chunk.

        Args:
            conditioning_tokens: Backbone hidden states used as cross-attention context.
            state: Current state, or None.

        Returns:
            Action chunk of shape ``[B, chunk_size, action_dim]``.
        """
        batch_size = conditioning_tokens.shape[0]
        actions = torch.randn(
            batch_size,
            self.action_horizon,
            self.config.action_dim,
            dtype=conditioning_tokens.dtype,
            device=conditioning_tokens.device,
        )
        dt = 1.0 / max(self.num_inference_timesteps, 1)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_value = int(t_cont * self.config.action_num_timestep_buckets)
            timesteps = torch.full(
                (batch_size,),
                t_value,
                device=conditioning_tokens.device,
                dtype=torch.long,
            )
            hidden_states = self._build_inputs(actions, state, timesteps)
            pred = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=conditioning_tokens,
                timestep=timesteps,
            )
            pred_velocity = self.action_decoder(pred[:, -self.action_horizon :])
            actions = actions + dt * pred_velocity
        return actions

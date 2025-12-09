# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Based on NVIDIA's GR00T implementation (Apache-2.0 licensed)
# Original source: https://github.com/NVIDIA/Isaac-GR00T

# ruff: noqa: FBT001, FBT002, ARG002, PLR0913, PLR0917, PLR2004
# ^^^^ Disabled because this module follows diffusers API conventions which use
# many boolean positional arguments and pass unused args to self.config.

"""Diffusion Transformer (DiT) for flow matching action head.

Uses HuggingFace diffusers components for attention and embeddings.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger(__name__)


def _import_diffusers() -> tuple:
    """Lazy import of diffusers components.

    Returns:
        Tuple of diffusers components.

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers import ConfigMixin, ModelMixin  # noqa: PLC0415
        from diffusers.configuration_utils import register_to_config  # noqa: PLC0415
        from diffusers.models.attention import Attention, FeedForward  # noqa: PLC0415
        from diffusers.models.embeddings import (  # noqa: PLC0415
            SinusoidalPositionalEmbedding,
            TimestepEmbedding,
            Timesteps,
        )
    except ImportError as e:
        msg = "DiT requires diffusers library.\n\nInstall with:\n    pip install diffusers"
        raise ImportError(msg) from e
    else:
        return (
            ConfigMixin,
            ModelMixin,
            register_to_config,
            Attention,
            FeedForward,
            SinusoidalPositionalEmbedding,
            TimestepEmbedding,
            Timesteps,
        )


class TimestepEncoder(nn.Module):
    """Encode diffusion timesteps into embeddings."""

    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32) -> None:
        """Initialize timestep encoder.

        Args:
            embedding_dim: Output embedding dimension.
            compute_dtype: Compute dtype (not used, for API compatibility).
        """
        super().__init__()
        _, _, _, _, _, _, timestep_embedding_cls, timesteps_cls = _import_diffusers()

        self.time_proj = timesteps_cls(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = timestep_embedding_cls(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode timesteps.

        Args:
            timesteps: Timestep indices.

        Returns:
            Timestep embeddings.
        """
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ) -> None:
        """Initialize adaptive layer normalization.

        Args:
            embedding_dim: Embedding dimension.
            norm_elementwise_affine: Whether to use affine params in norm.
            norm_eps: Epsilon for layer norm.
            chunk_dim: Dimension for chunking.
        """
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input tensor.
            temb: Timestep embedding.

        Returns:
            Normalized and scaled tensor.
        """
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Transformer block with optional cross-attention and adaptive normalization."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        """Initialize transformer block with attention and feed-forward layers.

        Raises:
            ValueError: If positional_embeddings is set but num_positional_embeddings is None.
        """
        super().__init__()
        (
            _config_mixin,
            _model_mixin,
            _register,
            attention_cls,
            feedforward_cls,
            sinusoidal_pos_embed_cls,
            _timestep_embed,
            _timesteps,
        ) = _import_diffusers()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and num_positional_embeddings is None:
            msg = "num_positional_embeddings must be defined if positional_embeddings is set"
            raise ValueError(msg)

        if positional_embeddings == "sinusoidal":
            self.pos_embed = sinusoidal_pos_embed_cls(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Self-attention with normalization
        self.norm1: AdaLayerNorm | nn.LayerNorm
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = attention_cls(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = feedforward_cls(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            hidden_states: Input tensor.
            attention_mask: Optional attention mask.
            encoder_hidden_states: Optional cross-attention context.
            encoder_attention_mask: Optional cross-attention mask.
            temb: Timestep embedding for adaptive norm.

        Returns:
            Processed hidden states.
        """
        # Self-attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


def _create_dit_class() -> type:
    """Factory to create DiT class with diffusers mixins.

    Returns:
        The DiT class with diffusers ConfigMixin and ModelMixin.
    """
    config_mixin, model_mixin, register_to_config, *_ = _import_diffusers()

    class DiT(model_mixin, config_mixin):  # type: ignore[misc,valid-type]
        """Diffusion Transformer for action generation.

        Cross-attention transformer conditioned on vision-language features
        and diffusion timesteps.
        """

        _supports_gradient_checkpointing = True

        @register_to_config
        def __init__(
            self,
            num_attention_heads: int = 8,
            attention_head_dim: int = 64,
            output_dim: int = 26,
            num_layers: int = 12,
            dropout: float = 0.1,
            attention_bias: bool = True,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: int | None = 1000,
            upcast_attention: bool = False,
            norm_type: str = "ada_norm",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5,
            max_num_positional_embeddings: int = 512,
            compute_dtype: torch.dtype = torch.float32,
            final_dropout: bool = True,
            positional_embeddings: str | None = "sinusoidal",
            interleave_self_attention: bool = False,
            cross_attention_dim: int | None = None,
        ) -> None:
            super().__init__()

            self.attention_head_dim = attention_head_dim
            self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            self.gradient_checkpointing = False

            # Timestep encoder
            self.timestep_encoder = TimestepEncoder(
                embedding_dim=self.inner_dim,
                compute_dtype=self.config.compute_dtype,
            )

            # Transformer blocks
            all_blocks = []
            for idx in range(self.config.num_layers):
                use_self_attn = idx % 2 == 1 and interleave_self_attention
                curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

                all_blocks.append(
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        activation_fn=self.config.activation_fn,
                        attention_bias=self.config.attention_bias,
                        upcast_attention=self.config.upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        positional_embeddings=positional_embeddings,
                        num_positional_embeddings=self.config.max_num_positional_embeddings,
                        final_dropout=final_dropout,
                        cross_attention_dim=curr_cross_attention_dim,
                    ),
                )
            self.transformer_blocks = nn.ModuleList(all_blocks)

            # Output blocks
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)

            logger.info(
                "DiT parameters: %d",
                sum(p.numel() for p in self.parameters() if p.requires_grad),
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            return_all_hidden_states: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Forward pass through DiT.

            Args:
                hidden_states: Action embeddings (B, T, D).
                encoder_hidden_states: VL features (B, S, D).
                timestep: Diffusion timestep.
                encoder_attention_mask: Optional attention mask.
                return_all_hidden_states: Whether to return intermediate states.

            Returns:
                Predicted velocity or tuple with intermediate states.
            """
            # Encode timesteps
            temb = self.timestep_encoder(timestep)

            hidden_states = hidden_states.contiguous()
            encoder_hidden_states = encoder_hidden_states.contiguous()

            all_hidden_states = [hidden_states]

            # Process through transformer blocks
            for idx, block in enumerate(self.transformer_blocks):
                if idx % 2 == 1 and self.config.interleave_self_attention:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        temb=temb,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=None,
                        temb=temb,
                    )
                all_hidden_states.append(hidden_states)

            # Output processing
            conditioning = temb
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

            if return_all_hidden_states:
                return self.proj_out_2(hidden_states), all_hidden_states
            return self.proj_out_2(hidden_states)

    return DiT


def _create_self_attention_transformer_class() -> type:
    """Factory to create SelfAttentionTransformer class with diffusers mixins.

    Returns:
        The SelfAttentionTransformer class with diffusers mixins.
    """
    config_mixin, model_mixin, register_to_config, *_ = _import_diffusers()

    class SelfAttentionTransformer(model_mixin, config_mixin):  # type: ignore[misc,valid-type]
        """Self-attention transformer for VL feature processing."""

        _supports_gradient_checkpointing = True

        @register_to_config
        def __init__(
            self,
            num_attention_heads: int = 8,
            attention_head_dim: int = 64,
            output_dim: int = 26,
            num_layers: int = 12,
            dropout: float = 0.1,
            attention_bias: bool = True,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: int | None = 1000,
            upcast_attention: bool = False,
            max_num_positional_embeddings: int = 512,
            compute_dtype: torch.dtype = torch.float32,
            final_dropout: bool = True,
            positional_embeddings: str | None = "sinusoidal",
            interleave_self_attention: bool = False,
        ) -> None:
            super().__init__()

            self.attention_head_dim = attention_head_dim
            self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            self.gradient_checkpointing = False

            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        activation_fn=self.config.activation_fn,
                        attention_bias=self.config.attention_bias,
                        upcast_attention=self.config.upcast_attention,
                        positional_embeddings=positional_embeddings,
                        num_positional_embeddings=self.config.max_num_positional_embeddings,
                        final_dropout=final_dropout,
                    )
                    for _ in range(self.config.num_layers)
                ],
            )

            logger.info(
                "SelfAttentionTransformer parameters: %d",
                sum(p.numel() for p in self.parameters() if p.requires_grad),
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            return_all_hidden_states: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
            """Forward pass through self-attention transformer.

            Args:
                hidden_states: Input tensor (B, T, D).
                return_all_hidden_states: Whether to return intermediate states.

            Returns:
                Processed hidden states.
            """
            hidden_states = hidden_states.contiguous()
            all_hidden_states = [hidden_states]

            for block in self.transformer_blocks:
                hidden_states = block(hidden_states)
                all_hidden_states.append(hidden_states)

            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

    return SelfAttentionTransformer


# Lazy class creation
_DiT: type | None = None
_SelfAttentionTransformer: type | None = None


def get_dit_class() -> type:
    """Get DiT class, creating it on first use.

    Returns:
        The DiT class.
    """
    global _DiT  # noqa: PLW0603
    if _DiT is None:
        _DiT = _create_dit_class()
    return _DiT


def get_self_attention_transformer_class() -> type:
    """Get SelfAttentionTransformer class, creating it on first use.

    Returns:
        The SelfAttentionTransformer class.
    """
    global _SelfAttentionTransformer  # noqa: PLW0603
    if _SelfAttentionTransformer is None:
        _SelfAttentionTransformer = _create_self_attention_transformer_class()
    return _SelfAttentionTransformer


# For direct imports (after diffusers is available)
def __getattr__(name: str) -> type:
    """Lazy attribute access for DiT and SelfAttentionTransformer.

    Args:
        name: Attribute name to access.

    Returns:
        The requested class (DiT or SelfAttentionTransformer).

    Raises:
        AttributeError: If the attribute is not found.
    """
    if name == "DiT":
        return get_dit_class()
    if name == "SelfAttentionTransformer":
        return get_self_attention_transformer_class()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

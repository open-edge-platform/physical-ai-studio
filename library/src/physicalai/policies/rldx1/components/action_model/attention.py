# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Self-attention transformer blocks (extracted from msat.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    # Only for static type checking; runtime import is lazy via _import_diffusers().
    from diffusers.models.embeddings import SinusoidalPositionalEmbedding


def _import_diffusers() -> tuple:
    """Lazy import of diffusers attention/embedding components.

    Returns:
        Tuple of (Attention, FeedForward, SinusoidalPositionalEmbedding).

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers.models.attention import Attention, FeedForward  # noqa: PLC0415
        from diffusers.models.embeddings import SinusoidalPositionalEmbedding  # noqa: PLC0415
    except ImportError as e:
        msg = "BasicTransformerBlock requires diffusers.\n\nInstall with:\n    pip install diffusers"
        raise ImportError(msg) from e
    else:
        return Attention, FeedForward, SinusoidalPositionalEmbedding


class BasicTransformerBlock(nn.Module):
    """Single transformer block used by the MSAT self-attention stack.

    The block applies layer norm, self-/cross-attention, residual connection,
    then feed-forward + residual connection.
    """

    pos_embed: SinusoidalPositionalEmbedding | None
    final_dropout: nn.Dropout | None

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",  # noqa: ARG002
        positional_embeddings: str | None = None,
        max_seq_length: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        """Initialize the transformer block.

        Args:
            dim: Hidden dimension of the input sequence.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Per-head feature dimension.
            dropout: Dropout probability used in attention/MLP modules.
            cross_attention_dim: Optional encoder dimension for cross-attention.
            activation_fn: Feed-forward activation function name.
            attention_bias: Whether attention projections use bias.
            upcast_attention: Whether to upcast attention computation for stability.
            norm_elementwise_affine: Whether LayerNorm uses learnable affine params.
            norm_type: Normalization type (kept for API compatibility).
            norm_eps: Epsilon for normalization layers.
            final_dropout: Whether to apply dropout after attention output.
            attention_type: Attention variant name (kept for API compatibility).
            positional_embeddings: Positional embedding mode. Supported: "sinusoidal" or None.
            max_seq_length: Maximum sequence length used by positional embeddings.
            ff_inner_dim: Optional feed-forward hidden dimension.
            ff_bias: Whether feed-forward linear layers use bias.
            attention_out_bias: Whether attention output projection uses bias.

        Raises:
            ValueError: If ``positional_embeddings`` is set without ``max_seq_length``,
                or if an unsupported ``positional_embeddings`` type is given.
        """
        super().__init__()
        attention_cls, feedforward_cls, sinusoidal_pos_embed_cls = _import_diffusers()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.max_seq_length = max_seq_length
        self.norm_type = norm_type

        if positional_embeddings and (max_seq_length is None):
            msg = "If `positional_embedding` type is defined, `max_seq_length` must also be defined."
            raise ValueError(msg)

        if positional_embeddings == "sinusoidal":
            if max_seq_length is None:
                msg = "If `positional_embedding` type is 'sinusoidal', `max_seq_length` must be defined."
                raise ValueError(msg)
            self.pos_embed = sinusoidal_pos_embed_cls(dim, max_seq_length=max_seq_length)
        elif positional_embeddings is None:
            self.pos_embed = None
        else:
            msg = "Invalid positional embedding type: `positional_embeddings` must be 'sinusoidal' or None."
            raise ValueError(msg)

        # 1. Self-Attn
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

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = feedforward_cls(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temb: torch.LongTensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run one transformer block pass.

        Args:
            hidden_states: Input tensor of shape ``(B, T, D)`` (or temporary
                4-D shape produced by upstream ops).
            attention_mask: Optional mask broadcastable to attention scores.
            encoder_hidden_states: Optional cross-attention context.
            temb: Optional timestep embedding (unused, kept for compatibility).

        Returns:
            Updated hidden states with the same semantic shape as input,
            typically ``(B, T, D)``.
        """
        # 0. Self-Attention
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
        if hidden_states.ndim == 4:  # noqa: PLR2004
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:  # noqa: PLR2004
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

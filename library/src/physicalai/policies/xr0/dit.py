# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rectified-flow action model for the XR0 Vision-Language-Action policy.

This module contains the *self-contained* DiT action head

The rectified-flow model (:class:`XR0FlowModel`) is deliberately
**VLM-independent**: the Qwen3-VL backbone outputs are passed in as
arguments rather than computed here, so this unit can be proven in isolation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Beta, LogisticNormal
from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, rotate_half

# A 4D rotary embedding carries a leading MRoPE section axis that is squeezed
# out before it is applied to the query/key tensors.
_ROTARY_4D_NDIM = 4

# ============================================================
# Helper functions
# ============================================================


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: shift-scale transformation used in DiT.

    Returns ``x * (1 + scale) + shift``, following the DiT / AdaLN-Zero
    formulation that lets the network learn whether to skip or amplify
    each sub-layer.

    Returns:
        The modulated tensor ``x * (1 + scale) + shift``.
    """
    return x * (1 + scale) + shift


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA).

    Args:
        hidden_states: ``(batch, num_kv_heads, seq_len, head_dim)``
        n_rep: Number of repetitions per KV head (``num_q_heads // num_kv_heads``).

    Returns:
        Tensor with shape ``(batch, num_q_heads, seq_len, head_dim)``.
    """
    _batch, _num_key_value_heads, _slen, _head_dim = hidden_states.shape
    return hidden_states.repeat_interleave(n_rep, dim=1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,  # noqa: ARG001
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor of shape ``(B, H, S, D)``.
        k: Key tensor of shape ``(B, H, S, D)``.
        cos: Cosine component of the rotary embedding.
        sin: Sine component of the rotary embedding.
        position_ids: Unused, kept for API compatibility.
        unsqueeze_dim: Dimension along which to unsqueeze cos/sin for broadcast.

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================
# Projectors & Embedders
# ============================================================


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector with optional GELU activation.

    Used to project between different dimensional spaces (e.g. state/action
    dimensions to DiT hidden size).

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        num_layers: Number of linear layers (intermediate layers use GELU).
        bias: Whether to include bias in linear layers.
    """

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, *, bias: bool = False) -> None:
        """Initialize the MLP projector."""
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.num_layers = num_layers

        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=bias)]
        for _ in range(1, num_layers):
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(output_dim, output_dim, bias=bias)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x`` through the MLP.

        Returns:
            The projected tensor.
        """
        return self.layers(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP.

    Used for conditioning the DiT on the diffusion timestep *t*.

    Args:
        hidden_size: Output dimension of the MLP (matches DiT hidden size).
        frequency_embedding_size: Dimension of the sinusoidal frequency embedding.
        dtype: Data type for the frequency embedding computation.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the timestep embedder."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Compute sinusoidal timestep embedding.

        Args:
            t: Timestep tensor of shape ``(B,)``.
            dim: Embedding dimension (should equal ``frequency_embedding_size``).
            max_period: Controls the frequency range of the embedding.

        Returns:
            Embedding tensor of shape ``(B, dim)``.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep *t* and return with a sequence dimension.

        Args:
            t: Timestep tensor of shape ``(B,)``.

        Returns:
            Embedding of shape ``(B, 1, hidden_size)``.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        # Add sequence dimension: (B, 1, D)
        return t_emb[:, None]


# ============================================================
# DiT components
# ============================================================


class DiTAttention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and VLM KV-cache for DiT decoder.

    Cross-attends to the VLM's cached key-value pairs while applying
    QK-RMSNorm for training stability.

    Args:
        hidden_size: Total attention dimension (``num_heads * head_dim``).
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads (grouped-query attention).
        dropout: Attention dropout probability (only active during training).
    """

    def __init__(self, hidden_size: int = 768, head_dim: int = 64, kv_heads: int = 2, dropout: float = 0.0) -> None:
        """Initialize the DiT cross-attention layer."""
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.kv_group = self.num_heads // kv_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_norm = Qwen2RMSNorm(self.head_dim)
        self.k_norm = Qwen2RMSNorm(self.head_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: QKV projection -> RoPE -> cross-attend with cached KV -> output.

        Args:
            hidden_state: Input of shape ``(B, S, D)``.
            past_key_values: Tuple of (cached_key, cached_value) from the VLM.
            position_embeds: (cos, sin) rotary embedding tensors.
            attn_mask: Boolean attention mask.

        Returns:
            Attended output of shape ``(B, S, D)``.
        """
        bsz, q_len, _ = hidden_state.size()

        qkv = self.qkv_proj(hidden_state)
        qkv = qkv.view(bsz, q_len, 3, self.num_heads, self.head_dim)
        query_states, key_states, value_states = qkv.unbind(2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary position embedding
        cos, sin = position_embeds
        if cos.ndim == _ROTARY_4D_NDIM:
            cos = cos[0]
            sin = sin[0]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Prepend cached KV from VLM
        k_cache, v_cache = past_key_values
        k_cache = repeat_kv(k_cache, self.kv_group)
        v_cache = repeat_kv(v_cache, self.kv_group)

        key_states = torch.cat([k_cache, key_states], dim=-2)
        value_states = torch.cat([v_cache, value_states], dim=-2)

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DiTMLP(nn.Module):
    """SwiGLU MLP used in DiT decoder layers.

    Args:
        hidden_size: Input and output dimension.  Intermediate size is ``4 * hidden_size``.
    """

    def __init__(self, hidden_size: int = 768) -> None:
        """Initialize the SwiGLU MLP."""
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: ``down(gelu(gate(x)) * up(x))``.

        Returns:
            The SwiGLU MLP output.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DecoderLayer(nn.Module):
    """DiT decoder layer with AdaLN modulation conditioned on diffusion timestep.

    Each layer produces 6 modulation parameters (shift/scale/gate for both
    the attention and FFN sub-layers) from the timestep embedding.

    Args:
        hidden_size: Model hidden dimension.
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads for GQA.
    """

    def __init__(self, hidden_size: int = 768, head_dim: int = 64, kv_heads: int = 2) -> None:
        """Initialize the DiT decoder layer."""
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = DiTAttention(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads)
        self.mlp = DiTMLP(hidden_size=hidden_size)

        # LayerNorms: input -> attn -> middle, post -> ffn -> final
        self.input_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.middle_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.post_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)
        self.final_layernorm = Qwen2RMSNorm(self.hidden_size, eps=1e-06)

        # AdaLN: produces 6 modulation parameters (shift/scale/gate for attn & ffn)
        self.adaln_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        t_embeds: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN modulation.

        Args:
            hidden_states: ``(B, S, D)`` input.
            past_key_values: VLM KV-cache for joint self-attention.
            position_embeds: (cos, sin) rotary embedding tensors.
            t_embeds: Timestep modulation parameters ``(B, 6, D)``.
            attn_mask: Boolean attention mask.

        Returns:
            Modulated output of shape ``(B, S, D)``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + t_embeds).chunk(
            6,
            dim=1,
        )

        # Attention block with AdaLN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_msa, scale_msa)
        hidden_states = self.attn(hidden_states, past_key_values, position_embeds, attn_mask=attn_mask)
        hidden_states = residual + gate_msa * hidden_states
        hidden_states = self.middle_layernorm(hidden_states)

        # FFN block with AdaLN
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = modulate(hidden_states, shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + gate_mlp * hidden_states
        return self.final_layernorm(hidden_states)


class DiT(nn.Module):
    """Diffusion Transformer that cross-attends to VLM KV-cache with AdaLN timestep conditioning.

    The DiT layers align with the *tail* of the VLM's KV-cache so that
    deeper DiT layers attend to later VLM layers.

    Args:
        hidden_size: Model hidden dimension.
        layer_num: Number of decoder layers.
        head_dim: Dimension per attention head.
        kv_heads: Number of KV heads for GQA.
    """

    def __init__(self, hidden_size: int = 768, layer_num: int = 8, head_dim: int = 128, kv_heads: int = 2) -> None:
        """Initialize the DiT decoder stack."""
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads) for _ in range(layer_num)],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        t_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all DiT layers.

        Args:
            hidden_states: ``(B, S, D)`` input.
            past_key_values: Per-layer VLM KV-cache.
            attn_mask: Boolean attention mask.
            position_embeds: (cos, sin) rotary embedding tensors.
            t_embeds: Timestep modulation parameters ``(B, 6, D)``.

        Returns:
            Output of shape ``(B, S, D)``.
        """
        # Align DiT layers with the tail of VLM KV-cache
        start_idx = max(0, len(past_key_values) - self.layer_num)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start_idx + i],
                position_embeds,
                t_embeds,
                attn_mask=attn_mask,
            )
        return hidden_states


class XR0FlowModel(nn.Module):
    """DiT action expert with rectified-flow generation (VLM features passed in)."""

    def __init__(
        self,
        state_shape: tuple[int, int] = (1, 32),
        action_shape: tuple[int, int] = (30, 32),
        dit_num_layers: int = 16,
        dit_hidden_size: int = 1024,
        dit_head_dim: int = 128,
        dit_kv_heads: int = 8,
        num_steps: int = 5,
        flow_sampling: str = "beta",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Build the DiT action expert, projectors, and flow-sampling distributions."""
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dit_num_layers = dit_num_layers
        self.dit_hidden_size = dit_hidden_size
        self.num_steps = num_steps

        # Rectified flow timestep sampling distributions.
        self.flow_sampling = flow_sampling
        self.logistic_normal = LogisticNormal(0.0, 1.0)
        self.beta = Beta(1.5, 1.0)

        # DiT policy head.
        self.dit = DiT(
            hidden_size=dit_hidden_size,
            kv_heads=dit_kv_heads,
            layer_num=dit_num_layers,
            head_dim=dit_head_dim,
        )

        # State / action projectors.
        self.state_projector = MLPProjector(
            input_dim=state_shape[-1],
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_projector = MLPProjector(
            input_dim=action_shape[-1],
            output_dim=dit_hidden_size,
            num_layers=2,
        )
        self.action_output_layer = MLPProjector(
            input_dim=dit_hidden_size,
            output_dim=action_shape[-1],
            num_layers=2,
        )

        # Timestep embedding for the diffusion t.
        self.t_embedder = TimestepEmbedder(dit_hidden_size, dtype=dtype)
        self.t_projector = MLPProjector(input_dim=dit_hidden_size, output_dim=6 * dit_hidden_size, bias=True)

        # Sink token prepended to the DiT input.
        self.sink = nn.Embedding(1, dit_hidden_size)

        self.to(dtype)

    # --------------------------------------------------------
    # Rectified flow methods
    # --------------------------------------------------------

    @torch.no_grad()
    def _sample_timestep(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Sample random timesteps for rectified flow training.

        The distribution is controlled by ``self.flow_sampling``:
        - ``"logit_normal"``: LogisticNormal(0, 1)
        - ``"beta"``: Beta(1.5, 1.0) rescaled to (0, 0.999)
        - otherwise: Uniform(0, 1)

        Args:
            batch_size: Number of timesteps to sample.
            dtype: Output tensor dtype.
            device: Output tensor device.

        Returns:
            Timestep tensor of shape ``(batch_size,)``.
        """
        if self.flow_sampling == "logit_normal":
            u = self.logistic_normal.sample((batch_size,))[:, 0].to(device)  # pyrefly: ignore[unsupported-operation]
        elif self.flow_sampling == "beta":
            u = self.beta.sample((batch_size,)).to(device)
            u = (1 - u) * 0.999
        else:
            u = torch.rand(size=(batch_size,), device=device)
        return u.to(dtype)

    @staticmethod
    @torch.no_grad()
    def _flow_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between noise and data: ``z_t = (1-t)*x0 + t*x1``.

        Args:
            x0: Noise sample (source distribution).
            x1: Data sample (target distribution).
            t: Interpolation coefficient in [0, 1].

        Returns:
            Interpolated tensor with the same shape as x0/x1.
        """
        return (1 - t) * x0 + t * x1

    @staticmethod
    @torch.no_grad()
    def _flow_velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Velocity target for rectified flow: ``v = x1 - x0``.

        Args:
            x0: Noise sample.
            x1: Data sample.

        Returns:
            Velocity tensor of the same shape.
        """
        return x1 - x0

    @torch.no_grad()
    def _flow_generate(self, x0: torch.Tensor, dit_kwargs: dict[str, Any]) -> torch.Tensor:
        """Euler integration: generate action from noise over ``num_steps`` steps.

        Args:
            x0: Initial noise tensor of shape ``(B, action_len, action_dim)``.
            dit_kwargs: Keyword arguments forwarded to ``dit_forward``.

        Returns:
            Denoised action prediction.
        """
        dt = 1.0 / self.num_steps
        z = x0.clone()
        for step in range(self.num_steps):
            t = torch.ones((z.shape[0], 1, 1), device=z.device, dtype=z.dtype) * step / self.num_steps
            v = self.dit_forward(z, t, **dit_kwargs)
            z += v * dt
        return z

    # --------------------------------------------------------
    # DiT forward
    # --------------------------------------------------------

    def dit_forward(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        action_mask: torch.Tensor,
        state_embed: torch.Tensor,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor,
        prefix_length: int = 0,
    ) -> torch.Tensor:
        """Single forward pass of DiT.

        1. Embed timestep *t* -> 6 AdaLN modulation parameters per layer.
        2. Project noisy action to hidden dim.
        3. Prepend [sink, state] tokens.
        4. Run DiT decoder layers (cross-attending to VLM KV-cache).
        5. Extract action tokens and project back to action dim.

        Args:
            noisy_action: Noisy action tensor ``(B, action_len, action_dim)``.
            t: Timestep ``(B, 1, 1)``.
            action_mask: Binary mask ``(B, action_len, action_dim)``.
            state_embed: Projected state ``(B, state_len, D)``.
            position_embeds: (cos, sin) rotary embeddings for DiT tokens.
            past_key_values: Per-layer VLM KV-cache.
            attn_mask: Boolean attention mask.
            prefix_length: Number of leading action tokens forced to zero.

        Returns:
            Predicted velocity (training) or action (inference) of shape
            ``(B, action_len, action_dim)``.
        """
        # Timestep conditioning: embed t -> 6 modulation parameters per layer.
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        # Project noisy action to DiT hidden dim.
        noisy_action *= action_mask
        noisy_action = self.action_projector(noisy_action)

        # Concatenate: [sink, state, noisy_action].
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        # DiT forward with VLM KV-cache (joint self-attention).
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        # Extract action tokens and project back to action dim.
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)
        if prefix_length > 0:
            output[:, :prefix_length] = 0.0

        return output

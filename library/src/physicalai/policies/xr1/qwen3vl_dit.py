# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DiT action expert for the XR1 policy.

The action expert is a Diffusion-Transformer stack that runs alongside the
Qwen3-VL backbone in a Mixture-of-Transformers layout: DiT layer ``i`` attends
over its own query tokens *and* over the key/value cache produced by VLM layer
``i``. The DiT therefore has the same depth as the backbone but a narrower hidden
size, which is what keeps inference affordable.

Ported from the reference implementation (``mibot/models/VLA/XR1.py``) with these
changes:

* every dimension is a constructor argument rather than a module-level default;
* timestep embeddings follow the module dtype instead of a hardcoded
  ``bfloat16`` cast, so ``float32`` runs and ONNX/OpenVINO tracing work;
* attention masks are accepted as boolean tensors and validated.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

MROPE_RANK = 4
RMS_NORM_EPS = 1e-6


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions, as rotary embeddings require.

    Implemented locally rather than imported from ``transformers.models.qwen2``:
    that is another model's private module, and depending on it version-locks the
    action expert to a particular ``transformers`` release. The maths is fixed by
    the RoPE formulation, and ``tests/unit/policies/xr1/test_qwen3vl_dit.py``
    asserts bit-for-bit agreement with the library implementation.

    Args:
        x: Tensor whose last dimension is even.

    Returns:
        Tensor of the same shape with the halves swapped and the first negated.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, computed in float32 for stability.

    Equivalent to ``RMSNorm``; see :func:`rotate_half` for why it is local.
    """

    def __init__(self, hidden_size: int, eps: float = RMS_NORM_EPS) -> None:
        """Initialize the norm.

        Args:
            hidden_size: Width of the normalized dimension.
            eps: Added to the variance for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension.

        Args:
            hidden_states: Input tensor.

        Returns:
            Normalized tensor in the input dtype.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Deliberately out-of-place: this norm is applied to views produced by
        # ``qkv.unbind(2)``, and autograd forbids in-place writes to such views.
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # noqa: PLR6104
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """Describe the norm in module printouts.

        Returns:
            Shape and epsilon.
        """
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation used by the feed-forward block.

    Args:
        x: Input tensor.

    Returns:
        ``x * sigmoid(x)``.
    """
    return F.silu(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply an adaptive layer-norm modulation.

    Args:
        x: Normalized hidden states.
        shift: Additive term broadcast over the sequence.
        scale: Multiplicative term broadcast over the sequence.

    Returns:
        Modulated hidden states.
    """
    return x * (1 + scale) + shift


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads.

    Args:
        hidden_states: Tensor of shape ``(batch, kv_heads, seq, head_dim)``.
        n_rep: Number of query heads per key/value head.

    Returns:
        Tensor of shape ``(batch, kv_heads * n_rep, seq, head_dim)``.
    """
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


def repeat_batch(hidden_states: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Expand a cached tensor to the requested batch size.

    Training repeats each sample several times so it can be denoised at multiple
    timesteps, while the VLM cache is computed once per sample.

    Args:
        hidden_states: Tensor whose first dimension divides ``batch_size``.
        batch_size: Target batch size.

    Returns:
        Tensor with first dimension equal to ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is not a multiple of the current batch.
    """
    current = hidden_states.shape[0]
    if current == batch_size:
        return hidden_states
    if batch_size % current != 0:
        msg = f"Cannot repeat batch of size {current} to {batch_size}"
        raise ValueError(msg)
    return hidden_states.repeat_interleave(batch_size // current, dim=0)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        query: Tensor of shape ``(batch, heads, seq, head_dim)``.
        key: Tensor of shape ``(batch, kv_heads, seq, head_dim)``.
        cos: Cosine embedding of shape ``(batch, seq, head_dim)``.
        sin: Sine embedding of shape ``(batch, seq, head_dim)``.

    Returns:
        The rotated query and key tensors.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class MLPProjector(nn.Module):
    """Small MLP used to project states, actions and timesteps."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inter_dim: int | None = None,
        num_layers: int = 1,
        *,
        bias: bool = False,
    ) -> None:
        """Initialize the projector.

        Args:
            input_dim: Input width.
            output_dim: Output width.
            inter_dim: Hidden width; defaults to ``output_dim``.
            num_layers: Number of linear layers. ``1`` is a plain projection.
            bias: Whether the linear layers carry a bias.

        Raises:
            ValueError: If ``num_layers`` is not positive.
        """
        super().__init__()
        if num_layers < 1:
            msg = f"num_layers ({num_layers}) must be positive"
            raise ValueError(msg)

        inter_dim = output_dim if inter_dim is None else inter_dim
        if num_layers == 1:
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim, bias=bias)]
        else:
            layers = [nn.Linear(input_dim, inter_dim, bias=bias)]
            for _ in range(1, num_layers - 1):
                layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, inter_dim, bias=bias)])
            layers.extend([nn.GELU(approximate="tanh"), nn.Linear(inter_dim, output_dim, bias=bias)])

        self.layers = nn.Sequential(*layers)
        self.apply(init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``x``.

        Args:
            x: Input tensor with last dimension ``input_dim``.

        Returns:
            Tensor with last dimension ``output_dim``.
        """
        return self.layers(x)


def init_linear_weights(module: nn.Module) -> None:
    """Initialize linear and RMS-norm weights the way the reference does.

    Args:
        module: Module to initialize in place.
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, RMSNorm):
        module.weight.data.fill_(1.0)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize the embedder.

        Args:
            hidden_size: Output width.
            frequency_embedding_size: Width of the sinusoidal features. Must be
                even.

        Raises:
            ValueError: If ``frequency_embedding_size`` is odd.
        """
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            msg = f"frequency_embedding_size ({frequency_embedding_size}) must be even"
            raise ValueError(msg)

        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed a batch of timesteps.

        Args:
            timestep: Tensor of shape ``(batch,)``.

        Returns:
            Tensor of shape ``(batch, 1, hidden_size)``.
        """
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half,
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Follow the module dtype rather than hardcoding bfloat16, so that
        # float32 runs and graph export both work.
        weight = cast("torch.Tensor", self.mlp[0].weight)
        embedding = embedding.to(weight.dtype)
        return self.mlp(embedding)[:, None]


class DiTAttention(nn.Module):
    """Self-attention over the DiT query tokens plus the VLM key/value cache."""

    def __init__(self, hidden_size: int = 1024, head_dim: int = 128, kv_heads: int = 8) -> None:
        """Initialize the attention block.

        Args:
            hidden_size: DiT hidden width.
            head_dim: Attention head dimension; must match the VLM head dim.
            kv_heads: Number of key/value heads in the VLM cache.

        Raises:
            ValueError: If the head geometry is inconsistent.
        """
        super().__init__()
        if hidden_size % head_dim != 0:
            msg = f"hidden_size ({hidden_size}) must be divisible by head_dim ({head_dim})"
            raise ValueError(msg)

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        if self.num_heads % kv_heads != 0:
            msg = f"num_heads ({self.num_heads}) must be divisible by kv_heads ({kv_heads})"
            raise ValueError(msg)
        self.kv_group = self.num_heads // kv_heads

        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend over the query tokens and the cached VLM keys and values.

        Args:
            hidden_states: Query tokens of shape ``(batch, query_len, hidden)``.
            past_key_values: ``(key, value)`` from the matching VLM layer, each of
                shape ``(cache_batch, kv_heads, cache_len, head_dim)``.
            position_embeds: ``(cos, sin)`` rotary embeddings for the query.
            attn_mask: Boolean mask of shape
                ``(batch, 1, query_len, cache_len + query_len)``.

        Returns:
            Tensor of shape ``(batch, query_len, hidden)``.
        """
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        cos, sin = position_embeds
        if cos.ndim == MROPE_RANK:
            cos, sin = cos[0], sin[0]
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        cache_key, cache_value = past_key_values
        cache_key = repeat_kv(repeat_batch(cache_key, batch_size), self.kv_group)
        cache_value = repeat_kv(repeat_batch(cache_value, batch_size), self.kv_group)
        key = torch.cat([cache_key.to(key.dtype), key], dim=-2)
        value = torch.cat([cache_value.to(value.dtype), value], dim=-2)

        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)


class DiTMLP(nn.Module):
    """Gated feed-forward block."""

    def __init__(self, hidden_size: int = 1024, intermediate_multiplier: int = 4) -> None:
        """Initialize the feed-forward block.

        Args:
            hidden_size: Hidden width.
            intermediate_multiplier: Expansion factor of the intermediate width.
        """
        super().__init__()
        intermediate_size = hidden_size * intermediate_multiplier
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform.

        Args:
            hidden_states: Input tensor.

        Returns:
            Tensor of the same shape as the input.
        """
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DiTDecoderLayer(nn.Module):
    """One DiT layer: adaptive layer norm, attention over the VLM cache, MLP."""

    _MODULATION_TERMS = 6

    def __init__(self, hidden_size: int = 1024, head_dim: int = 128, kv_heads: int = 8) -> None:
        """Initialize the layer.

        Args:
            hidden_size: DiT hidden width.
            head_dim: Attention head dimension.
            kv_heads: Number of key/value heads in the VLM cache.
        """
        super().__init__()
        self.attn = DiTAttention(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads)
        self.mlp = DiTMLP(hidden_size=hidden_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.adaln_table = nn.Parameter(
            torch.randn(self._MODULATION_TERMS, hidden_size) / hidden_size**0.5,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: tuple[torch.Tensor, torch.Tensor],
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        timestep: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the layer.

        Args:
            hidden_states: Query tokens of shape ``(batch, query_len, hidden)``.
            past_key_values: ``(key, value)`` from the matching VLM layer.
            position_embeds: ``(cos, sin)`` rotary embeddings for the query.
            timestep: Timestep conditioning of shape ``(batch, 6, hidden)``.
            attn_mask: Boolean attention mask.

        Returns:
            Tensor of shape ``(batch, query_len, hidden)``.
        """
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (self.adaln_table[None] + timestep).chunk(
            self._MODULATION_TERMS,
            dim=1,
        )

        residual = hidden_states
        hidden_states = modulate(self.input_layernorm(hidden_states), shift_attn, scale_attn)
        hidden_states = residual + gate_attn * self.attn(
            hidden_states,
            past_key_values,
            position_embeds,
            attn_mask,
        )

        residual = hidden_states
        hidden_states = modulate(self.post_layernorm(hidden_states), shift_mlp, scale_mlp)
        return residual + gate_mlp * self.mlp(hidden_states)


class DiT(nn.Module):
    """Stack of DiT layers reading the VLM key/value cache layer by layer."""

    def __init__(
        self,
        hidden_size: int = 1024,
        layer_num: int = 36,
        head_dim: int = 128,
        kv_heads: int = 8,
    ) -> None:
        """Initialize the stack.

        Args:
            hidden_size: DiT hidden width.
            layer_num: Number of DiT layers.
            head_dim: Attention head dimension; must match the VLM head dim.
            kv_heads: Number of key/value heads in the VLM cache.
        """
        super().__init__()
        self.layer_num = layer_num
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [DiTDecoderLayer(hidden_size=hidden_size, head_dim=head_dim, kv_heads=kv_heads) for _ in range(layer_num)],
        )
        self.apply(init_linear_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        attn_mask: torch.Tensor | None,
        position_embeds: tuple[torch.Tensor, torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run every DiT layer against its matching VLM cache layer.

        The DiT is aligned to the *last* ``layer_num`` layers of the VLM cache, so
        a shallower DiT reads the deepest VLM representations.

        Args:
            hidden_states: Query tokens of shape ``(batch, query_len, hidden)``.
            past_key_values: Per-layer ``(key, value)`` pairs from the VLM.
            attn_mask: Boolean attention mask.
            position_embeds: ``(cos, sin)`` rotary embeddings for the query.
            timestep: Timestep conditioning of shape ``(batch, 6, hidden)``.

        Returns:
            Tensor of shape ``(batch, query_len, hidden)``.

        Raises:
            ValueError: If the cache has fewer layers than the DiT.
        """
        start = len(past_key_values) - self.layer_num
        if start < 0:
            msg = (
                f"VLM cache has {len(past_key_values)} layers, fewer than the DiT's "
                f"{self.layer_num}; reduce dit_num_layers or use a deeper backbone"
            )
            raise ValueError(msg)

        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                past_key_values[start + index],
                position_embeds,
                timestep,
                attn_mask,
            )
        return hidden_states

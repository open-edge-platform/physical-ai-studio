# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 flow-matching action expert.

Checkpoint key prefix: ``model.action_expert.*``. Denoises an action
trajectory conditioned on per-layer text KV context via cross-attention. Each
block self-attends over the action horizon, cross-attends into one text layer's
KV, and is modulated by a sinusoidal timestep embedding (DiT-style).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .rms_norm import RMSNorm

if TYPE_CHECKING:
    from collections.abc import Sequence

KVContext = tuple[torch.Tensor, torch.Tensor]


def _round_up_multiple(value: int, multiple_of: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple_of``.

    Returns:
        The smallest integer greater than or equal to ``value`` that is
        a multiple of ``multiple_of``. If ``multiple_of`` is less than
        or equal to zero, returns ``value`` unchanged.
    """
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply DiT-style ``shift``/``scale`` modulation over the sequence.

    Returns:
        The modulated tensor, of the same shape as ``x``.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


@dataclass
class ActionExpertContext:
    """Precomputed per-step context shared across denoising steps."""

    kv_contexts: Sequence[KVContext]
    cross_mask: torch.Tensor | None
    self_mask: torch.Tensor | None
    valid_action: torch.Tensor | None
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None


class ActionExpertRMSNorm(RMSNorm):
    """RMS norm, optionally without a learnable weight (matches checkpoint)."""

    def __init__(self, size: int, *, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        """Build the norm, registering a weight only when affine."""
        super().__init__(size, eps=eps, elementwise_affine=elementwise_affine)


class ActionExpertRotaryEmbedding(nn.Module):
    """Rotary embedding for the action-expert self-attention."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        """Store rotary parameters (``head_dim`` must be even).

        Raises:
            ValueError: If ``head_dim`` is not even.
        """
        super().__init__()
        if head_dim % 2 != 0:
            msg = "RoPE requires an even head_dim."
            raise ValueError(msg)
        self.head_dim = head_dim
        self.base = base

    def build_cache(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the ``(cos, sin)`` cache for a given sequence length.

        Returns:
            A tuple ``(cos, sin)`` of tensors, each of shape
            ``(1, 1, seq_len, head_dim // 2)``, containing the cosine
            and sine values used to rotate ``q``/``k``.
        """
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype).view(1, 1, seq_len, half_dim)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to ``q`` and ``k``.

        Returns:
            A tuple ``(q_rotated, k_rotated)`` of tensors, each the same
            shape as the corresponding input, with rotary position
            embeddings applied.
        """
        cos, sin = rope_cache
        half_dim = self.head_dim // 2

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of flow-matching timesteps."""

    def __init__(self, dim: int) -> None:
        """Store the embedding dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a ``(batch,)`` timestep vector into ``(batch, dim)``.

        Returns:
            A tensor of shape ``(batch, dim)`` containing the
            sinusoidal timestep embeddings. If ``dim`` is odd, the
            result is zero-padded by one column to match ``dim``.
        """
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1)),
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ActionExpertSelfAttention(nn.Module):
    """Self-attention over the action horizon with QK-norm and rotary."""

    def __init__(self, hidden_size: int, num_heads: int, *, qk_norm: bool, qk_norm_eps: float, rope: bool) -> None:
        """Build the fused QKV / output projections and optional norms/rotary."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Run masked self-attention over the action sequence.

        Returns:
            The attention output, a tensor of shape
            ``(batch, seq_len, hidden)``.
        """
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.rope is not None and rope_cache is not None:
            q, k = self.rope(q, k, rope_cache)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out)


class ActionExpertCrossAttention(nn.Module):
    """Cross-attention from action tokens into one text layer's KV context."""

    def __init__(self, hidden_size: int, num_heads: int, *, qk_norm: bool, qk_norm_eps: float) -> None:
        """Build the query/output projections and optional QK norms."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend action tokens into ``(kv_k, kv_v)`` context heads.

        Returns:
            Cross attended action tokens with KV.
        """
        batch, tgt_len, hidden = x.shape
        q = self.q_proj(x).view(batch, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = kv_k.transpose(1, 2)
        v = kv_v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(batch, tgt_len, hidden)
        return self.out_proj(out)


class ActionExpertMLP(nn.Module):
    """SwiGLU feed-forward for an action-expert block."""

    def __init__(self, hidden_size: int, *, mlp_ratio: float, multiple_of: int) -> None:
        """Build the up/gate/down projections."""
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transform.

        Returns:
            Projection of action expert.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionExpertModulation(nn.Module):
    """Produce DiT modulation parameters from the timestep conditioning."""

    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        """Build the modulation projection producing ``num_chunks`` vectors."""
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Project the activated conditioning into modulation parameters.

        Returns:
            Projected and SiLu'd output of DiT.
        """
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    """Self-attention + cross-attention + MLP block with timestep modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        qk_norm: bool,
        qk_norm_eps: float,
        rope: bool,
    ) -> None:
        """Build the norms, attentions, MLP and modulation projection."""
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(hidden_size, num_heads, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps)
        self.mlp = ActionExpertMLP(hidden_size, mlp_ratio=mlp_ratio, multiple_of=ffn_multiple_of)
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_kv: KVContext,
        self_attn_mask: torch.Tensor | None,
        cross_attn_mask: torch.Tensor | None,
        is_causal: bool,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Apply modulated self-attn, cross-attn and MLP with residuals.

        Returns:
            Self attention, cross attention fo action expert.
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(conditioning).chunk(9, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.self_attn(  # noqa: PLR6104
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
            rope_cache=rope_cache,
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(  # noqa: PLR6104
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=cross_attn_mask,
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.ff_norm(x), shift_mlp, scale_mlp))


class ActionExpertFinalLayer(nn.Module):
    """Final modulated projection from hidden states to action velocities."""

    def __init__(self, hidden_size: int, output_dim: int) -> None:
        """Build the norm, modulation and output projection."""
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Modulate and project to the action velocity space.

        Returns:
            Ouptut of hidden states, project to action space.
        """
        shift, scale = self.modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class ActionExpert(nn.Module):
    """Per-layer cross-attending denoiser for continuous action generation."""

    def __init__(
        self,
        *,
        max_action_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        ffn_multiple_of: int,
        timestep_embed_dim: int,
        context_layer_norm: bool,
        qk_norm: bool,
        qk_norm_eps: float,
        rope: bool,
        causal_attn: bool,
        llm_kv_dim: int,
        llm_num_layers: int,
    ) -> None:
        """Build time/action embeddings, KV projections, blocks and final layer.

        Raises:
            ValueError: if action action expers have no block per text layer.
        """
        super().__init__()
        if num_layers != llm_num_layers:
            msg = f"Action expert needs one block per text layer: {num_layers} != {llm_num_layers}."
            raise ValueError(msg)
        self.num_heads = num_heads
        self.action_head_dim = hidden_size // num_heads
        self.causal_attn = causal_attn
        # Toggled by :meth:`gradient_checkpointing_enable` so each block is
        # recomputed during the backward pass to trade compute for memory. Has
        # no effect outside of training (``self.training`` and
        # ``torch.is_grad_enabled()`` are both required).
        self.gradient_checkpointing: bool = False

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.action_embed = nn.Linear(max_action_dim, hidden_size)
        self.context_k_proj = nn.Linear(llm_kv_dim, hidden_size, bias=False)
        self.context_v_proj = nn.Linear(llm_kv_dim, hidden_size, bias=False)
        self.context_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6) if context_layer_norm else nn.Identity()
        self.blocks = nn.ModuleList([
            ActionExpertBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                ffn_multiple_of=ffn_multiple_of,
                qk_norm=qk_norm,
                qk_norm_eps=qk_norm_eps,
                rope=rope,
            )
            for _ in range(num_layers)
        ])
        self.final_layer = ActionExpertFinalLayer(hidden_size, max_action_dim)

    def time_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps into the modulation conditioning vector.

        Returns:
            The condition vectors from timesteps.
        """
        conditioning = self.time_embed[0](timesteps).to(self.time_embed[1].weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _project_kv(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """Project text KV to the action head layout ``(batch, seq, heads, hd)``.

        Returns:
            KV projected.
        """
        flat = self.context_norm(proj(x))
        return flat.view(flat.shape[0], flat.shape[1], self.num_heads, self.action_head_dim)

    def project_kv_context(
        self,
        block: ActionExpertBlock,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> KVContext:
        """Project one text layer's KV states for its matching action block.

        Returns:
            Projected key and value tensors in the action-head layout.
        """
        key_context = self._project_kv(key_states, self.context_k_proj)
        value_context = self._project_kv(value_states, self.context_v_proj)
        if block.cross_attn.k_norm is not None:
            key_context = block.cross_attn.k_norm(key_context.transpose(1, 2)).transpose(1, 2)
        return key_context, value_context

    def prepare_context_metadata(
        self,
        *,
        encoder_attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """Build the masks and rotary cache shared by all action-expert layers.

        Returns:
            Cross-attention mask, self-attention mask, and rotary cache.
        """
        cross_mask = None
        if encoder_attention_mask is not None:
            valid = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
            cross_mask = (1.0 - valid) * torch.finfo(dtype).min

        self_mask = None
        if self.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)
            self_mask = causal[None, None].to(dtype) * torch.finfo(dtype).min

        rope_cache = None
        if len(self.blocks) > 0:
            first_block = cast("ActionExpertBlock", self.blocks[0])
            rope = first_block.self_attn.rope
            if rope is not None:
                rope_cache = rope.build_cache(seq_len=seq_len, device=device, dtype=dtype)
        return cross_mask, self_mask, rope_cache

    @staticmethod
    def expand_context_for_flow_timesteps(
        context: ActionExpertContext,
        num_flow_timesteps: int,
    ) -> ActionExpertContext:
        """Repeat a per-example context along the batch dim for multi-sample flow training.

        When ``config.num_flow_timesteps > 1``, several independent
        (timestep, noise) samples are drawn per training example to reduce
        the flow-matching loss's variance (see
        :meth:`MolmoAct2Backbone.predict_flow_velocity`). The text/vision
        encoder still runs only once per example; this repeats its per-layer
        KV context and cross-attention mask ``num_flow_timesteps`` times
        (interleaved, matching how the denoising inputs are flattened via
        ``actions.repeat_interleave(num_flow_timesteps, dim=0)``) so the
        action expert can process the flattened
        ``(batch * num_flow_timesteps,)`` batch. ``self_mask`` and
        ``rope_cache`` only depend on the action horizon, not the batch, so
        they are reused unchanged.

        Returns:
            A new :class:`ActionExpertContext` with per-example tensors
            repeated ``num_flow_timesteps`` times along the batch dim.
        """
        if num_flow_timesteps == 1:
            return context
        kv_contexts = [
            (
                k_ctx.repeat_interleave(num_flow_timesteps, dim=0),
                v_ctx.repeat_interleave(num_flow_timesteps, dim=0),
            )
            for k_ctx, v_ctx in context.kv_contexts
        ]
        cross_mask = (
            context.cross_mask.repeat_interleave(num_flow_timesteps, dim=0) if context.cross_mask is not None else None
        )
        valid_action = (
            context.valid_action.repeat_interleave(num_flow_timesteps, dim=0)
            if context.valid_action is not None
            else None
        )
        return ActionExpertContext(
            kv_contexts=kv_contexts,
            cross_mask=cross_mask,
            self_mask=context.self_mask,
            valid_action=valid_action,
            rope_cache=context.rope_cache,
        )

    def prepare_context(
        self,
        *,
        encoder_kv_states: Sequence[KVContext],
        encoder_attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActionExpertContext:
        """Project text KV per layer and build the attention masks and rope cache.

        Returns:
            ActionExpertContext per step context shared across de-nosing steps.
        """
        kv_contexts: list[KVContext] = []
        for block, (k_in, v_in) in zip(self.blocks, encoder_kv_states, strict=False):
            kv_contexts.append(self.project_kv_context(cast("ActionExpertBlock", block), k_in, v_in))
        cross_mask, self_mask, rope_cache = self.prepare_context_metadata(
            encoder_attention_mask=encoder_attention_mask,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

        return ActionExpertContext(
            kv_contexts=kv_contexts,
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=None,
            rope_cache=rope_cache,
        )

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
    ) -> torch.Tensor:
        """Predict the flow velocity for a single denoising step.

        Returns:
            Flow velocity for one de-noising step.
        """
        conditioning = self.time_conditioning(timesteps)
        x = self.action_embed(actions)
        use_gradient_checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block, kv_context in zip(self.blocks, context.kv_contexts, strict=False):
            if use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    conditioning,
                    cross_kv=kv_context,
                    self_attn_mask=context.self_mask,
                    cross_attn_mask=context.cross_mask,
                    is_causal=self.causal_attn,
                    rope_cache=context.rope_cache,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    conditioning,
                    cross_kv=kv_context,
                    self_attn_mask=context.self_mask,
                    cross_attn_mask=context.cross_mask,
                    is_causal=self.causal_attn,
                    rope_cache=context.rope_cache,
                )
        return self.final_layer(x, conditioning)

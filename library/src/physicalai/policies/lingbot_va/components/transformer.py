# Copyright 2024-2025 The Robbyant Team Authors.
# Copyright 2026 The HuggingFace Inc. team.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dual-stream Wan2.2 transformer backbone used by LingBot-VA.

The backbone is a "mixture of transformers": a video-latent stream
(``patch_embedding_mlp`` -> ``blocks`` -> ``proj_out``) and an action stream
(``action_embedder`` -> ``blocks`` -> ``action_proj_out``) share the same transformer
blocks and the same text conditioning, but keep separate input/output projections and
timestep embedders.

Two forward paths are implemented:

- :meth:`WanTransformer3DModel.forward` — one denoising step for one stream, with KV
  caching. This is the streaming inference path.
- :meth:`WanTransformer3DModel.forward_train` — both streams packed into one sequence with
  block-causal flex-attention masks. This is the flow-matching training path.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn

from .attention import FlexAttnFunc, WanAttention, WanRotaryPosEmbed

_PACK_ALIGNMENT = 128
"""Flex attention packs the training sequence to a multiple of this many tokens."""


def _lazy_import_diffusers() -> tuple[Any, Any, Any, Any]:
    """Import the diffusers building blocks the Wan backbone reuses.

    Returns:
        Tuple of ``(FeedForward, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps)``.

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers.models.attention import FeedForward  # noqa: PLC0415
        from diffusers.models.embeddings import (  # noqa: PLC0415
            PixArtAlphaTextProjection,
            TimestepEmbedding,
            Timesteps,
        )
    except ImportError as e:
        msg = "LingBot-VA requires diffusers.\n\nInstall with:\n    uv pip install 'physicalai-train[lingbot_va]'"
        raise ImportError(msg) from e
    return FeedForward, PixArtAlphaTextProjection, TimestepEmbedding, Timesteps


def _lazy_import_fp32_layer_norm() -> type[nn.Module]:
    """Import ``FP32LayerNorm`` from diffusers.

    Returns:
        The ``FP32LayerNorm`` class.

    Raises:
        ImportError: If diffusers is not installed.
    """
    try:
        from diffusers.models.normalization import FP32LayerNorm  # noqa: PLC0415
    except ImportError as e:
        msg = "LingBot-VA requires diffusers.\n\nInstall with:\n    uv pip install 'physicalai-train[lingbot_va]'"
        raise ImportError(msg) from e
    return FP32LayerNorm


def data_seq_to_patch(
    patch_size: tuple[int, int, int],
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """Reshape a flattened patch sequence back into a ``[B, C, F, H, W]`` latent grid.

    Args:
        patch_size: Latent patch size ``(t, h, w)``.
        data_seq: Flattened patch sequence of shape ``[B, F*H*W, C]``.
        latent_num_frames: Number of latent frames ``F``.
        latent_height: Latent height ``H``.
        latent_width: Latent width ``W``.
        batch_size: Batch size of ``data_seq``.

    Returns:
        Latent grid of shape ``[B, C, F, H, W]``.
    """
    p_t, p_h, p_w = patch_size
    data_patch = data_seq.reshape(
        batch_size,
        latent_num_frames // p_t,
        latent_height // p_h,
        latent_width // p_w,
        p_t,
        p_h,
        p_w,
        -1,
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def get_mesh_id(
    f: int,
    h: int,
    w: int,
    t: int,
    f_w: int = 1,
    f_shift: int = 0,
    *,
    action: bool = False,
) -> torch.Tensor:
    """Build the ``(frame, height, width, stream)`` grid ids that index the rotary embedding.

    Action tokens carry ``-1`` on the height/width axes and a fractional offset on the
    frame axis, which keeps them ordered within their frame while staying distinguishable
    from video-latent tokens.

    Args:
        f: Number of frames.
        h: Grid height (number of action sub-steps per frame in action mode).
        w: Grid width (``1`` in action mode).
        t: Stream id written to the fourth row.
        f_w: Frame-index stride.
        f_shift: Frame-index offset of the first frame.
        action: Whether these ids belong to the action stream.

    Returns:
        Grid-id tensor of shape ``[4, f*h*w]``.
    """
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    ff, hh, ww = torch.meshgrid(f_idx, torch.arange(h), torch.arange(w), indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        # Out-of-place: meshgrid returns expanded views, which reject in-place writes.
        ff = ff + ff_offset  # noqa: PLR6104
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)


class WanTimeTextImageEmbedding(nn.Module):
    """Timestep and text conditioning embedder.

    Args:
        dim: Model dimension.
        time_freq_dim: Width of the sinusoidal timestep features.
        time_proj_dim: Width of the per-block modulation projection (``6 * dim``).
        text_embed_dim: Width of the text-encoder hidden states.
    """

    def __init__(self, dim: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int) -> None:
        """Build the timestep projection stack and the text projection."""
        super().__init__()
        feed_forward, text_projection, timestep_embedding, timesteps = _lazy_import_diffusers()
        del feed_forward

        self.timesteps_proj = timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = timestep_embedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = text_projection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed per-token timesteps.

        Args:
            timestep: Timesteps of shape ``[B, S]``.
            dtype: Dtype to cast the embedding to.

        Returns:
            Tuple of ``(temb, timestep_proj)`` with shapes ``[B, S, dim]`` and
            ``[B, S, 6 * dim]``.
        """
        b, seq_len = timestep.shape
        timestep = self.timesteps_proj(timestep.reshape(-1))
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if time_embedder_dtype not in {timestep.dtype, torch.int8}:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(b, seq_len, -1), timestep_proj.reshape(b, seq_len, -1)


class WanTransformerBlock(nn.Module):
    """One Wan2.2 transformer block: self-attention, text cross-attention, feed-forward.

    Args:
        dim: Model dimension.
        ffn_dim: Feed-forward inner dimension.
        num_heads: Number of attention heads.
        cross_attn_norm: Whether to layer-norm before cross-attention.
        eps: Layer-norm / RMS-norm epsilon.
        attn_mode: Attention backend, see :class:`~.attention.WanAttention`.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        *,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        attn_mode: str = "torch",
    ) -> None:
        """Build the attention, cross-attention and feed-forward sub-layers."""
        super().__init__()
        feed_forward, _, _, _ = _lazy_import_diffusers()
        fp32_layer_norm = _lazy_import_fp32_layer_norm()
        self.attn_mode = attn_mode

        self.norm1 = fp32_layer_norm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = fp32_layer_norm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.ffn = feed_forward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = fp32_layer_norm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor | None,
        update_cache: int = 0,
        cache_name: str = "pos",
    ) -> torch.Tensor:
        """Run the block.

        Args:
            hidden_states: Token features of shape ``[B, S, C]``.
            encoder_hidden_states: Projected text features of shape ``[B, S_text, C]``.
            temb: Per-token modulation of shape ``[B, S, 6, C]``.
            rotary_emb: Rotary factors for the self-attention positions.
            update_cache: KV-cache commit mode, see :meth:`~.attention.WanAttention.forward`.
            cache_name: Name of the KV cache pool.

        Returns:
            Updated token features of shape ``[B, S, C]``.
        """
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            chunk.squeeze(1) for chunk in rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(6, dim=1)
        )

        norm_hidden_states = (self.norm1(hidden_states.float()) * (1.0 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + attn_output  # noqa: PLR6104 - out-of-place keeps autograd valid

        norm_hidden_states = (self.norm3(hidden_states.float()) * (1.0 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states,
        )
        ff_output = self.ffn(norm_hidden_states)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


class WanTransformer3DModel(nn.Module):
    """Dual-stream (video + action) Wan2.2 DiT backbone with autoregressive KV caching.

    Args:
        patch_size: Latent patch size ``(t, h, w)``.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Per-head dimension.
        in_channels: Video-latent channels consumed by the patch embedder.
        out_channels: Video-latent channels produced by ``proj_out``.
        action_dim: Width of the multi-embodiment action vector.
        text_dim: Width of the text-encoder hidden states.
        freq_dim: Width of the sinusoidal timestep features.
        ffn_dim: Feed-forward inner dimension.
        num_layers: Number of transformer blocks.
        cross_attn_norm: Whether blocks layer-norm before cross-attention.
        eps: Layer-norm epsilon.
        rope_max_seq_len: Maximum rotary sequence length.
        attn_mode: Attention backend; ``"flex"`` is required for training.
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        action_dim: int = 30,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        *,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        attn_mode: str = "torch",
    ) -> None:
        """Build the shared blocks plus the per-stream embedders and output heads."""
        super().__init__()
        fp32_layer_norm = _lazy_import_fp32_layer_norm()

        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(in_channels * math.prod(patch_size), inner_dim)
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                inner_dim,
                ffn_dim,
                num_attention_heads,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                attn_mode=attn_mode,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = fp32_layer_norm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    @property
    def self_attentions(self) -> list[WanAttention]:
        """The blocks' self-attention modules, which own the streaming KV caches."""
        return [block.attn1 for block in cast("list[WanTransformerBlock]", list(self.blocks))]

    def clear_cache(self, cache_name: str) -> None:
        """Drop every block's KV cache pool for ``cache_name``."""
        for attention in self.self_attentions:
            attention.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name: str) -> None:
        """Invalidate every block's predicted (not yet observed) cache entries."""
        for attention in self.self_attentions:
            attention.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        """Allocate the streaming KV cache sized for ``attn_window`` chunks.

        Args:
            cache_name: Name of the cache pool.
            attn_window: Attention window in chunks (half video, half action).
            latent_token_per_chunk: Video-latent tokens produced per chunk.
            action_token_per_chunk: Action tokens produced per chunk.
            device: Device to allocate on.
            dtype: Dtype of the cached keys/values.
            batch_size: Batch size (2 under classifier-free guidance).
        """
        total_token_len = (attn_window // 2) * latent_token_per_chunk + (attn_window // 2) * action_token_per_chunk
        for attention in self.self_attentions:
            attention.init_kv_cache(
                cache_name,
                total_token_len,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _input_embed(self, latents: torch.Tensor, input_type: str = "latent") -> torch.Tensor:
        """Embed one stream's raw input into token features.

        Args:
            latents: Video latents ``[B, C, F, H, W]``, actions ``[B, C, F, N, 1]`` or
                text embeddings ``[B, S, text_dim]``.
            input_type: One of ``"latent"``, ``"action"``, ``"text"``.

        Returns:
            Token features of shape ``[B, S, inner_dim]``.

        Raises:
            ValueError: If ``input_type`` is unknown.
        """
        if input_type == "latent":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            return self.patch_embedding_mlp(hidden_states)
        if input_type == "action":
            return self.action_embedder(rearrange(latents, "b c f h w -> b (f h w) c"))
        if input_type == "text":
            return self.condition_embedder.text_embedder(latents)
        msg = f"Unsupported input type: {input_type}"
        raise ValueError(msg)

    def _time_embed(
        self,
        timesteps: torch.Tensor,
        h: int,
        w: int,
        dtype: torch.dtype,
        *,
        action_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast per-frame timesteps to per-token modulation features.

        Args:
            timesteps: Per-frame timesteps of shape ``[B, F]``.
            h: Spatial height of the stream being embedded.
            w: Spatial width of the stream being embedded.
            dtype: Dtype of the returned embeddings.
            action_mode: Whether the action stream (no spatial patching) is embedded.

        Returns:
            Tuple of ``(temb, timestep_proj)``; ``timestep_proj`` is ``[B, S, 6, C]``.
        """
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(timesteps, (h // patch_scale_h) * (w // patch_scale_w), dim=1)
        condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = condition_embedder(latent_time_steps, dtype=dtype)
        return temb, timestep_proj.unflatten(2, (6, -1))

    def _modulate_out(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply the final adaptive layer-norm modulation.

        Args:
            hidden_states: Token features of shape ``[B, S, C]``.
            temb: Per-token conditioning of shape ``[B, S, C]``.

        Returns:
            Modulated features of shape ``[B, S, C]``.
        """
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        return (self.norm_out(hidden_states.float()) * (1.0 + scale) + shift).type_as(hidden_states)

    def forward_train(self, input_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: PLR0914
        """Dual-stream training forward (flow matching). Requires ``attn_mode='flex'``.

        Both streams are packed into a single sequence together with their clean
        (conditioning) counterparts, and the block-causal flex-attention masks decide who
        may attend to whom.

        Args:
            input_dict: Dict with ``latent_dict`` / ``action_dict`` (each holding
                ``noisy_latents``, ``latent``, ``timesteps``, ``cond_timesteps``,
                ``grid_id``, ``text_emb``), plus ``chunk_size`` and ``window_size``.

        Returns:
            Tuple of ``(latent_prediction, action_prediction)``.
        """
        latent_dict, action_dict = input_dict["latent_dict"], input_dict["action_dict"]
        for stream in (latent_dict, action_dict):
            stream["noisy_latents"] = stream["noisy_latents"].to(torch.bfloat16)
            stream["latent"] = stream["latent"].to(torch.bfloat16)

        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = self._input_embed(latent_dict["noisy_latents"], "latent").flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict["noisy_latents"], "action").flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], "text").flatten(0, 1)[None]
        cond_latent_hidden_states = self._input_embed(latent_dict["latent"], "latent").flatten(0, 1)[None]
        cond_action_hidden_states = self._input_embed(action_dict["latent"], "action").flatten(0, 1)[None]

        hidden_states = torch.cat(
            [latent_hidden_states, cond_latent_hidden_states, action_hidden_states, cond_action_hidden_states],
            dim=1,
        )

        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        rotary_emb = self.rope(torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2))[:, :, None]

        latent_time_steps = torch.cat([
            latent_dict["timesteps"].flatten(0, 1),
            latent_dict["cond_timesteps"].flatten(0, 1),
        ])[None]
        action_time_steps = torch.cat([
            action_dict["timesteps"].flatten(0, 1),
            action_dict["cond_timesteps"].flatten(0, 1),
        ])[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict["noisy_latents"].shape[-2],
            action_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        padded_length = (_PACK_ALIGNMENT - hidden_states.shape[1] % _PACK_ALIGNMENT) % _PACK_ALIGNMENT
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            cond_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            cond_action_hidden_states.shape[1],
            padded_length,
        ]

        FlexAttnFunc.init_mask(
            latent_dict["noisy_latents"].shape,
            action_dict["noisy_latents"].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
            text_seq_len=latent_dict["text_emb"].shape[1],
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, text_hidden_states, timestep_proj, rotary_emb, update_cache=0)

        hidden_states = self._modulate_out(hidden_states, temb)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)

        latent_out = self.proj_out(latent_hidden_states)
        latent_out = rearrange(
            latent_out,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_out = rearrange(self.action_proj_out(action_hidden_states), "1 (b l) c -> b l c", b=batch_size)
        return latent_out, action_out

    def forward(
        self,
        input_dict: dict[str, Any],
        update_cache: int = 0,
        cache_name: str = "pos",
        *,
        action_mode: bool = False,
        train_mode: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one denoising step for one stream (or delegate to the training pass).

        Args:
            input_dict: Stream dict with ``noisy_latents``, ``timesteps``, ``grid_id`` and
                ``text_emb`` (or the training dict when ``train_mode`` is set).
            update_cache: KV-cache commit mode, see :meth:`~.attention.WanAttention.forward`.
            cache_name: Name of the KV cache pool.
            action_mode: Whether the action stream is being denoised.
            train_mode: Delegate to :meth:`forward_train`.

        Returns:
            The stream's velocity prediction, or the ``(latent, action)`` pair in
            training mode.
        """
        if train_mode:
            return self.forward_train(input_dict)

        input_type = "action" if action_mode else "latent"
        hidden_states = self._input_embed(input_dict["noisy_latents"], input_type)
        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])
        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]

        temb, timestep_proj = self._time_embed(
            input_dict["timesteps"],
            input_dict["noisy_latents"].shape[-2],
            input_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=action_mode,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        hidden_states = self._modulate_out(hidden_states, temb)

        if action_mode:
            return self.action_proj_out(hidden_states)
        return rearrange(self.proj_out(hidden_states), "b l (n c) -> b (l n) c", n=math.prod(self.patch_size))

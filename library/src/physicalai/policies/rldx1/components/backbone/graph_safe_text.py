# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe Qwen3-VL text model (LLM decoder + VTC compression) for export.

Ports ``rldx/inference/backbone/llm/model/graph_safe_qwen3vl_text_model.py``
from RLWRLD/RLDX-1 for the v1 export path (SDPA attention, no KV cache).

The eager decoder wraps each layer in a ``LayerWrapper`` that, at the
``internal_projection`` layer, compresses the image-token span via
``torch.nonzero`` + a data-dependent ``.item()`` -- untraceable. Because
``input_ids`` (and thus the image span) is fixed for a given export,
:func:`_find_compress_info` resolves the begin/end indices **once, eagerly**,
into Python ints, and :class:`GraphSafeQwen3VLTextModel.forward` performs the
same compression with static slices. The trace ``input_ids`` may carry random
or dummy task tokens, so a caller can pass a canonical, contract-derived
``compress_input_ids`` (the left-padded, right-aligned ``[vision][suffix]``
layout the runtime token composer produces) to resolve the span against the
true runtime positions instead of the traced tokens. SDPA attention with
``attention_mask=None`` is causal (the attention sets ``is_causal=True``), so
no data-dependent mask is built.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

_POSITION_IDS_NDIM_2D = 2
_MROPE_AXES = 3


def _compute_fa_kwargs(seq_len: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Static flash-attention varlen kwargs for one contiguous sequence.

    Returns:
        A tuple ``(cu_seqlens, max_seqlen)`` for one contiguous sequence.
    """
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return cu, int(seq_len)


def _find_compress_info(
    language_model: nn.Module,
    input_ids: torch.Tensor,
    n_cog_tokens: int,
    num_views: int | None = None,
) -> dict[str, int] | None:
    """Resolve the VTC compression layer and its static begin/end indices.

    Runs ``LayerWrapper.get_removing_indices`` once, eagerly, and materializes
    the begin/end positions as Python ints.

    Args:
        language_model: The decoder whose ``.layers`` are ``LayerWrapper``s.
        input_ids: The fixed export token ids (with the image-token span).
        n_cog_tokens: Number of appended cognition tokens.
        num_views: Camera-view count. ``num_views >= 2`` keeps the last
            ``num_views - 1`` image sets uncompressed. When ``begin >= end`` the
            vanilla path skips compression, so this returns ``None``.

    Returns:
        A dict with ``compress_layer_idx`` / ``static_begin`` / ``static_end`` /
        ``static_out_len``, or ``None`` when no compression happens.
    """
    language_model_any = cast("Any", language_model)
    for idx, layer in enumerate(language_model_any.layers):
        if (
            hasattr(layer, "layer")
            and hasattr(layer, "internal_projection")
            and layer.layer_idx == layer.internal_projection
        ):
            with torch.no_grad():
                dummy = torch.zeros(1, input_ids.shape[1], 1, device=input_ids.device)
                num_views_list = [int(num_views)] if num_views is not None else None
                begin_idx, end_idx = layer.get_removing_indices(dummy, input_ids, num_views=num_views_list)
            begin = int(begin_idx[0, 0].item())
            end = int(end_idx[0, 0].item())
            if begin >= end:
                return None
            length_llm = input_ids.shape[1] + n_cog_tokens
            out_len = begin + 1 + (length_llm - end)
            return {
                "compress_layer_idx": idx,
                "static_begin": begin,
                "static_end": end,
                "static_out_len": out_len,
            }
    return None


class GraphSafeQwen3VLTextModel(nn.Module):
    """Qwen3-VL text decoder with static VTC compression and no KV cache.

    Data-dependent ops replaced:
      - ``LayerWrapper`` compression (``torch.nonzero`` + ``.item()``) -> static
        begin/end slice with a mean cognition token.
      - causal-mask construction -> ``attention_mask=None`` (SDPA ``is_causal``).
    """

    def __init__(
        self,
        text_model: nn.Module,
        input_ids: torch.Tensor,
        n_cog_tokens: int = 0,
        attn_impl: str = "sdpa",
        num_views: int | None = None,
        compress_input_ids: torch.Tensor | None = None,
    ) -> None:
        """Initialize the graph-safe text decoder wrapper.

        Args:
            text_model: Wrapped eager Qwen3-VL decoder.
            input_ids: Fixed export token ids.
            n_cog_tokens: Number of appended cognition tokens.
            attn_impl: Attention backend name.
            num_views: Number of camera views for VTC span logic.
            compress_input_ids: Optional canonical ids used to resolve the
                static compression span.

        Raises:
            ValueError: If ``compress_input_ids`` length differs from
                ``input_ids`` length.
        """
        super().__init__()
        self._text_model = text_model
        self.attn_impl = attn_impl

        length_ids = input_ids.shape[1]
        device = input_ids.device
        length_pre = length_ids + n_cog_tokens

        # The VTC span is resolved by locating the 151652 image-pad markers in the
        # ids. Trace ids may carry random/dummy tokens, so resolve against a
        # canonical, contract-derived layout when provided: the runtime prompt is
        # left-padded and right-aligns [vision x num_images][suffix], so the span
        # is fixed regardless of the (possibly random) traced tokens.
        compress_ids = input_ids if compress_input_ids is None else compress_input_ids.to(device)
        if compress_ids.shape[1] != length_ids:
            msg = (
                f"compress_input_ids length {compress_ids.shape[1]} must match input_ids "
                f"length {length_ids} so static VTC indices stay aligned."
            )
            raise ValueError(msg)
        self.compress_info = _find_compress_info(text_model, compress_ids, n_cog_tokens, num_views=num_views)
        length_post = self.compress_info["static_out_len"] if self.compress_info is not None else length_pre

        # Static FA varlen params (Python int + constant tensor; no runtime .item()).
        # Register the tensors as buffers so torch.export treats them as buffers
        # rather than lifted constants (non-persistent buffers are lifted as
        # constants, which trips a fake-tensor check).
        pre_cu, self.pre_max_seqlen = _compute_fa_kwargs(length_pre, device)
        post_cu, self.post_max_seqlen = _compute_fa_kwargs(length_post, device)
        nn.Module.register_buffer(self, "pre_cu_seqlens", pre_cu)
        nn.Module.register_buffer(self, "post_cu_seqlens", post_cu)

    def forward(  # noqa: PLR0914
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        deepstack_add: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        """Run the decoder with static compression.

        Args:
            input_ids: Token ids (used only when ``inputs_embeds`` is ``None``).
            inputs_embeds: Precomputed input embeddings ``(B, L, D)``.
            position_ids: ``(B, L)`` or ``(3, B, L)`` M-RoPE ids.
            position_embeddings: Optional precomputed ``(cos, sin)``; computed
                from ``position_ids`` when omitted.
            deepstack_add: Optional ``(num_ds, B, L, D)`` additive DeepStack
                features applied after the first ``num_ds`` layers.
            attention_mask: Optional ``(B, L)`` 1/0 pad mask (dynamic-prompt
                export). When given, a causal+pad additive mask is built and
                threaded through the layers; when ``None`` the layers rely on
                SDPA ``is_causal`` (static single-prompt export).

        Returns:
            ``BaseModelOutputWithPast`` with the final hidden states.
        """
        tm = cast("Any", self._text_model)

        if inputs_embeds is None:
            inputs_embeds = tm.embed_tokens(input_ids)

        if position_ids is not None and position_ids.ndim == _POSITION_IDS_NDIM_2D:
            position_ids = position_ids[None, ...].expand(_MROPE_AXES, position_ids.shape[0], -1)

        hidden_states = inputs_embeds
        if position_embeddings is None:
            position_embeddings = tm.rotary_emb(hidden_states, position_ids)

        pad_2d = attention_mask
        attn_mask_4d = (
            self._causal_pad_mask(pad_2d, hidden_states.dtype, hidden_states.device) if pad_2d is not None else None
        )

        ci = self.compress_info
        use_fa = self.attn_impl == "flash_attention_2"
        cu_seqlens = self.pre_cu_seqlens
        max_seqlen = self.pre_max_seqlen

        for idx, layer in enumerate(tm.layers):
            inner = layer.layer if hasattr(layer, "layer") else layer

            if ci is not None and idx == ci["compress_layer_idx"]:
                begin, end = ci["static_begin"], ci["static_end"]
                n_drop = end - begin
                drop_mask = torch.zeros(
                    1,
                    hidden_states.shape[1],
                    1,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                drop_mask[:, begin:end, :] = 1.0
                motion = (hidden_states * drop_mask).sum(dim=1, keepdim=True) / n_drop
                front = hidden_states[:, :begin, :]
                back = hidden_states[:, end:, :]
                hidden_states = torch.cat([front, motion, back], dim=1)

                cos, sin = position_embeddings
                cos = torch.cat([cos[:, :begin], cos[:, begin : begin + 1], cos[:, end:]], dim=1)
                sin = torch.cat([sin[:, :begin], sin[:, begin : begin + 1], sin[:, end:]], dim=1)
                position_embeddings = (cos, sin)

                # The image span is all real tokens, so the collapsed slot stays
                # attended; rebuild the mask for the shortened sequence.
                if pad_2d is not None:
                    pad_2d = torch.cat([pad_2d[:, :begin], pad_2d[:, begin : begin + 1], pad_2d[:, end:]], dim=1)
                    attn_mask_4d = self._causal_pad_mask(pad_2d, hidden_states.dtype, hidden_states.device)

                cu_seqlens = self.post_cu_seqlens
                max_seqlen = self.post_max_seqlen

            if use_fa:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                )
            else:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask_4d,
                )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            if deepstack_add is not None and idx < deepstack_add.shape[0]:
                hidden_states += deepstack_add[idx]

        hidden_states = tm.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    @staticmethod
    def _causal_pad_mask(pad_2d: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build a ``(B, 1, L, L)`` additive causal + padding attention mask.

        Args:
            pad_2d: ``(B, L)`` 1/0 mask (1 = real token).
            dtype: Additive-mask dtype (``0`` where attended, ``dtype.min`` else).
            device: Target device.

        Returns:
            Additive mask suitable for SDPA ``attn_mask``.
        """
        batch, length = pad_2d.shape
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
        keep = causal[None, None] & pad_2d[:, None, None, :].to(torch.bool)
        mask = torch.zeros(batch, 1, length, length, dtype=dtype, device=device)
        return mask.masked_fill(~keep, torch.finfo(dtype).min)

    def __getattr__(self, name: str) -> nn.Module | torch.Tensor:
        """Delegate unknown attributes to the wrapped text model.

        Returns:
            The attribute from this wrapper or the wrapped text model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return cast("nn.Module | torch.Tensor", getattr(self._text_model, name))

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO export workarounds for the XR0 self-contained IR.

Two OpenVINO-specific fixes the XR0 export needs
"""

from __future__ import annotations

import types
from typing import Any, Protocol

import torch

# Marker attribute used to make the RMSNorm install idempotent (re-running export
# prep in the same process must not double-wrap a module's forward).
_PATCHED_FLAG = "_ov_friendly_rmsnorm"


class _RMSNormLike(Protocol):
    """Structural type for the RMSNorm modules whose forward we swap."""

    weight: torch.Tensor
    variance_epsilon: float


def export_rmsnorm_forward(self: _RMSNormLike, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward that reduces over a positive, static axis.

    Drop-in replacement for the stock ``Qwen2RMSNorm`` / ``Qwen3VLTextRMSNorm``
    forward. Identical math, but the reduction axis is the concrete positive
    ``ndim - 1`` instead of ``-1`` so the OpenVINO PyTorch frontend emits a valid
    ``ReduceMean`` axis constant (a negative axis is mis-materialized and makes the
    exported IR fail to load).

    Args:
        self: The RMSNorm module (provides ``weight`` and ``variance_epsilon``).
        hidden_states: The input activations to normalize.

    Returns:
        The RMS-normalized, weight-scaled activations in the input dtype.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32).clone()
    axis = hidden_states.dim() - 1  # concrete positive int -> clean ReduceMean axis
    variance = hidden_states.pow(2).mean(axis, keepdim=True)
    hidden_states *= torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def _is_rmsnorm(module: torch.nn.Module) -> bool:
    """Return whether ``module`` is an RMSNorm to patch.

    Identified structurally (has ``variance_epsilon`` and ``weight``) and by class
    name suffix, so it matches both the Qwen2 (DiT head) and Qwen3-VL text
    RMSNorm variants without importing the ``transformers`` classes.

    Returns:
        ``True`` if the module is an RMSNorm whose forward should be swapped.
    """
    return (
        type(module).__name__.endswith("RMSNorm") and hasattr(module, "variance_epsilon") and hasattr(module, "weight")
    )


def install_export_rmsnorm(module: torch.nn.Module) -> int:
    """Swap every RMSNorm instance in ``module`` to the OpenVINO-friendly forward.

    Walks the whole submodule tree and, for each RMSNorm instance, rebinds its
    ``forward`` to :func:`ov_friendly_rmsnorm_forward`. Pass the top-level
    :class:`~physicalai.policies.xr0.model.XR0Model` to cover both the Qwen3-VL text
    backbone and the DiT action head in a single call. Idempotent: modules already
    patched are skipped, so it is safe to call more than once.

    Args:
        module: The model (or subtree) whose RMSNorm modules should be patched.

    Returns:
        The number of RMSNorm modules that were patched by this call.
    """
    patched = 0
    for submodule in module.modules():
        if not _is_rmsnorm(submodule) or getattr(submodule, _PATCHED_FLAG, False):
            continue
        submodule.forward = types.MethodType(export_rmsnorm_forward, submodule)
        submodule.__dict__[_PATCHED_FLAG] = True
        patched += 1
    return patched


# --------------------------------------------------------------------------- #
# Export-friendly reimplementations of the stock Qwen3-VL ops.                 #
#                                                                             #
# Each of these is numerically identical to a stock ``transformers`` op but   #
# expressed so ``torch.export`` / OpenVINO can convert it. They are           #
# module-level (not closures) so each can be unit-tested in isolation against #
# its stock counterpart; ``XR0Qwen3VL._ensure_export_patch`` installs them.   #
# --------------------------------------------------------------------------- #


def export_precompute_vision_geometry(visual: Any, image_grid_thw: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: ANN401
    """Precompute the vision tower's data-dependent geometry tensors off-graph.

    Since transformers 5.10 ``Qwen3VLVisionModel.forward`` builds its
    interpolation position-embedding indices/weights, rotary ``position_ids`` and
    ``cu_seqlens`` through the standalone ``transformers.vision_utils.get_vision_*``
    helpers, each of which iterates ``grid_thw.tolist()`` (untraceable under
    ``torch.export``). Every helper first pops a precomputed tensor from its
    ``kwargs`` when present, so the export path computes them once here from the
    fixed baked geometry (concrete ints, eager) and injects them back through the
    forward ``kwargs`` (see :meth:`XR0Qwen3VL._ensure_export_patch`). The injected
    values are exactly what the tower would compute itself, so the traced graph is
    numerically unchanged but free of the data-dependent ``.tolist()`` loops.

    Args:
        visual: The vision tower (reads ``num_grid_per_side`` / ``spatial_merge_size``
            and the ``interpolation_mode`` / ``interpolation_align_corners`` flags).
        image_grid_thw: The fixed vision geometry ``(num_images, 3)``.

    Returns:
        Mapping of the stock forward kwarg names (``position_ids``,
        ``interp_indices``, ``interp_weights``, ``cu_seqlens``) to their
        precomputed tensors.
    """
    from transformers.vision_utils import (  # noqa: PLC0415
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        image_grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size,
    )
    return {
        "position_ids": get_vision_position_ids(image_grid_thw, visual.spatial_merge_size),
        "interp_indices": interp_indices,
        "interp_weights": interp_weights,
        "cu_seqlens": get_vision_cu_seqlens(image_grid_thw),
    }


def export_vision_attn_forward(
    attn: Any,  # noqa: ANN401
    split_sizes: list[int],
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Export-friendly vision attention for one block.

    Numerically identical to stock ``Qwen3VLVisionAttention.forward`` (non-flash
    path), but it splits the per-image attention windows by the constant Python
    ``split_sizes`` instead of ``lengths.tolist()`` (derived from ``cu_seqlens``,
    which yields unbacked symints under ``torch.export``) and calls SDPA directly.
    The shared attention interface passes ``enable_gqa=True``, which the ONNX
    exporter rejects unless ``q_heads > kv_heads``; the vision tower has equal
    q/kv heads, so a plain SDPA is numerically identical.

    Args:
        attn: The vision attention module (reads ``qkv``, ``proj``, ``num_heads``,
            ``scaling``).
        split_sizes: Per-window token counts summing to the sequence length.
        hidden_states: ``(seq_len, dim)`` input hidden states.
        position_embeddings: The ``(cos, sin)`` rotary embeddings.

    Returns:
        The ``(seq_len, dim)`` attention output.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision  # noqa: PLC0415

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    splits = [torch.split(tensor, split_sizes, dim=2) for tensor in (query_states, key_states, value_states)]
    attn_outputs = [
        torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=attn.scaling,
        ).transpose(1, 2)
        for q, k, v in zip(*splits, strict=False)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return attn.proj(attn_output)


def export_image_token_gather(
    input_ids: torch.Tensor,
    image_token_id: int,
    num_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate the image tokens traceably (no ``nonzero`` / boolean ``Where``).

    Args:
        input_ids: Runtime token ids ``(1, L)``.
        image_token_id: The configured image-token id.
        num_image_tokens: The baked image-token count (gather clamp bound).

    Returns:
        ``(gather_index, image_mask)`` where ``image_mask`` is a ``(L,)`` long
        ``{0, 1}`` mask marking the image tokens and ``gather_index`` is a
        ``(L,)`` long tensor mapping each sequence position to its image-token
        rank (clamped to ``num_image_tokens``), for use with ``index_select``
        into the per image-token geometry / visual embeds.
    """
    image_mask = (input_ids.reshape(-1) == image_token_id).to(torch.long)
    image_rank = torch.cumsum(image_mask, dim=0) - 1
    gather_index = image_rank.clamp(min=0, max=num_image_tokens - 1)
    return gather_index, image_mask


def export_scatter_visual_embeds(
    inputs_embeds: torch.Tensor,
    image_gather_index: torch.Tensor,
    image_token_mask: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Merge visual embeds into token embeddings by masked gather.

    Export-friendly replacement for the stock image/text merge, which uses
    ``masked_scatter`` (-> an unconvertible ``Where`` whose operand shapes
    disagree). Each sequence position gathers its image-token embedding (``Gather``)
    and is blended in by the ``{0, 1}`` image mask; because the mask is exactly
    ``0`` / ``1`` the blend replaces the image slots and leaves the text slots
    untouched, numerically identical to the stock merge for a single-batch
    sequence.

    Args:
        inputs_embeds: ``(1, seq_len, hidden)`` token embeddings.
        image_gather_index: ``(seq_len,)`` per-position image-token rank (clamped)
            for ``index_select`` into ``image_embeds``.
        image_token_mask: ``(seq_len,)`` ``{0, 1}`` mask of the image tokens.
        image_embeds: ``(num_visual, hidden)`` visual embeddings.

    Returns:
        ``(1, seq_len, hidden)`` embeddings with the image slots replaced.
    """
    row = inputs_embeds[0]
    gathered = image_embeds.index_select(0, image_gather_index).to(row.dtype)
    mask = image_token_mask.to(row.dtype).unsqueeze(-1)
    merged = row * (1 - mask) + gathered * mask
    return merged.unsqueeze(0)


def export_add_deepstack_embeds(
    hidden_states: torch.Tensor,
    image_gather_index: torch.Tensor,
    image_token_mask: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    """Add deepstack visual features at the image-token positions by masked gather.

    Export-friendly replacement for the stock ``_deepstack_process``, which adds
    ``visual_embeds`` into ``hidden_states`` via boolean-mask assignment (-> an
    unconvertible ``Where``). Each position gathers its deepstack feature
    (``Gather``) and adds it in, scaled by the ``{0, 1}`` image mask, so only the
    image slots change -- numerically identical for a single-batch sequence.

    Args:
        hidden_states: ``(1, seq_len, hidden)`` decoder hidden states.
        image_gather_index: ``(seq_len,)`` per-position image-token rank (clamped)
            for ``index_select`` into ``visual_embeds``.
        image_token_mask: ``(seq_len,)`` ``{0, 1}`` mask of the image tokens.
        visual_embeds: ``(num_visual, hidden)`` deepstack features to add.

    Returns:
        ``(1, seq_len, hidden)`` hidden states with the features added.
    """
    row = hidden_states[0]
    gathered = visual_embeds.index_select(0, image_gather_index).to(row.dtype)
    mask = image_token_mask.to(row.dtype).unsqueeze(-1)
    updated = row + gathered * mask
    return updated.unsqueeze(0)


def export_mrope_position_ids(
    attention_mask: torch.Tensor,
    image_gather_index: torch.Tensor,
    image_token_mask: torch.Tensor,
    image_row: torch.Tensor,
    image_col: torch.Tensor,
    image_advance: torch.Tensor,
) -> torch.Tensor:
    """Recompute the 3D MRoPE ``position_ids`` traceably for one prompt.

    Reproduces stock ``get_rope_index`` for a single right-padded sequence from
    the baked image *geometry* (``image_row`` / ``image_col`` / ``image_advance``)
    and the runtime ``attention_mask`` plus the image-token gather, using
    cumulative sums and gathers only (no ``nonzero`` / boolean ``Where``), so it
    is captured by ``torch.export`` and lowers to OpenVINO-convertible ops.

    Args:
        attention_mask: Runtime attention mask ``(1, L)``.
        image_gather_index: ``(L,)`` per-position image-token rank (clamped) for
            ``index_select`` into the per image-token geometry.
        image_token_mask: ``(L,)`` ``{0, 1}`` mask of the image tokens.
        image_row: ``(num_image_tokens,)`` baked merged-grid row offset.
        image_col: ``(num_image_tokens,)`` baked merged-grid column offset.
        image_advance: ``(num_image_tokens,)`` baked per-block MRoPE advance
            (non-zero only at the last token of each image block).

    Returns:
        The 3D MRoPE ``position_ids`` tensor ``(3, 1, L)`` for this prompt.
    """
    valid = attention_mask.reshape(-1).to(torch.long)
    row = image_row.index_select(0, image_gather_index) * image_token_mask
    col = image_col.index_select(0, image_gather_index) * image_token_mask
    advance = image_advance.index_select(0, image_gather_index) * image_token_mask
    text_mask = valid * (1 - image_token_mask)
    # Per-token position increment: +1 per valid text token and the whole
    # image block's advance applied at its last token. The exclusive cumsum is
    # the block start position shared by every token of a block.
    step = text_mask + advance
    base = torch.cumsum(step, dim=0) - step
    temporal = base
    height = base + row
    width = base + col
    position_ids = torch.stack([temporal, height, width], dim=0).unsqueeze(1)
    # Padded tokens carry position 0, matching stock ``get_rope_index``.
    return position_ids * valid.reshape(1, 1, -1)


def export_build_additive_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Build a 4-D additive causal mask from a 2-D padding mask.

    Export-friendly replacement for the text model's default SDPA mask builder.
    The stock builder combines the causal and padding masks with a vmapped
    advanced index (``padding_mask[batch_idx, kv_idx]``), which lowers to a
    boolean ``GatherND`` the Intel GPU plugin has no kernel for. Passing an
    already-4-D mask makes ``create_causal_mask`` early-exit and return it as-is
    (see ``transformers.masking_utils._preprocess_mask_arguments``), so the gather
    is never emitted. This builds the same mask with pure broadcasting
    (comparisons + ``where`` -> ``Less``/``And``/``Select``, all convertible).

    Args:
        attention_mask: The 2-D padding mask ``(batch, seq_len)`` (1 = keep,
            0 = pad).
        dtype: The floating dtype of the attention scores; masked positions are
            filled with its most-negative value.

    Returns:
        A ``(batch, 1, seq_len, seq_len)`` additive mask (``0`` where attended,
        ``finfo(dtype).min`` where masked).
    """
    batch, seq_len = attention_mask.shape
    device = attention_mask.device
    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :] <= positions[:, None]  # (q, kv): kv <= q
    keep_kv = attention_mask.to(torch.bool).reshape(batch, 1, seq_len)  # valid key positions
    allowed = causal[None, :, :] & keep_kv  # (batch, q, kv)
    additive = torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)
    return additive.unsqueeze(1)

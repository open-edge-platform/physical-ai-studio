# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO export workarounds for the XR0 self-contained IR.

Two OpenVINO-specific fixes the XR0 export needs
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

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


# Rank-6 spatial layout ``(gh, gw, C, tp, ps_h, ps_w)`` obtained from the
# ``(tp, C, gh, ps_h, gw, ps_w)`` reshape.
_SPATIAL_PERM = (2, 4, 1, 0, 3, 5)
# Rank-5 merge-block reorder ``(gh/m, m_h, gw/m, m_w, D) -> (gh/m, gw/m, m_h, m_w, D)``.
_MERGE_PERM = (0, 2, 1, 3, 4)


def patchify_image_grid(
    images: torch.Tensor,
    grid_thw: Sequence[Sequence[int]],
    *,
    temporal_patch_size: int,
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Patchify a normalized image grid exactly like the Qwen3-VL image processor.

    Mirrors the transformers ``Qwen2VLImageProcessor._preprocess`` patchify: each
    single-frame image is temporally duplicated to ``temporal_patch_size`` frames,
    reshaped into merge-size patch blocks and transposed, then flattened to the flat
    ``pixel_values`` layout the vision tower consumes.

    Args:
        images: Normalized image grid of shape ``(num_images, C, H, W)`` where
            ``H == grid_h * patch_size`` and ``W == grid_w * patch_size`` for the
            matching ``grid_thw`` row.
        grid_thw: One ``(grid_t, grid_h, grid_w)`` triple per image (``grid_t`` is the
            temporal group count, ``1`` for a still image).
        temporal_patch_size: Number of frames grouped per temporal patch.
        patch_size: Spatial patch side length in pixels.
        merge_size: Spatial merge block size.

    Returns:
        The flat ``pixel_values`` tensor of shape
        ``(sum(grid_t * grid_h * grid_w), C * temporal_patch_size * patch_size ** 2)``.
    """
    flattened: list[torch.Tensor] = []
    for index, (grid_t, grid_h, grid_w) in enumerate(grid_thw):
        image = images[index : index + 1]  # (1, C, H, W)
        channel = image.shape[1]
        feature = channel * temporal_patch_size * patch_size * patch_size
        patches = image.repeat(temporal_patch_size, 1, 1, 1)  # (tp, C, H, W)
        # The transformers patchify is a single 9-D reshape/transpose, but the
        # Intel GPU plugin's tensor layouts top out at rank 6. Split it into
        # rank<=6 steps that produce a bit-identical result: first carve the
        # spatial patches (rank 6), then reorder the merge blocks (rank 5).
        patches = patches.reshape(
            temporal_patch_size,
            channel,
            grid_h,
            patch_size,
            grid_w,
            patch_size,
        )
        patches = patches.permute(*_SPATIAL_PERM)  # (gh, gw, C, tp, ps_h, ps_w)
        patches = patches.reshape(grid_h * grid_w, feature)
        patches = patches.reshape(
            grid_h // merge_size,
            merge_size,
            grid_w // merge_size,
            merge_size,
            feature,
        )
        patches = patches.permute(*_MERGE_PERM)
        flattened.append(patches.reshape(grid_t * grid_h * grid_w, feature))
    return torch.cat(flattened, dim=0)


# --------------------------------------------------------------------------- #
# Export-friendly reimplementations of the stock Qwen3-VL ops.                 #
#                                                                             #
# Each of these is numerically identical to a stock ``transformers`` op but   #
# expressed so ``torch.export`` / OpenVINO can convert it. They are           #
# module-level (not closures) so each can be unit-tested in isolation against #
# its stock counterpart; ``XR0Qwen3VL._ensure_export_patch`` installs them.   #
# --------------------------------------------------------------------------- #


def export_rot_pos_emb(visual: Any, grid_thw_list: list[list[int]]) -> torch.Tensor:  # noqa: PLR0914, ANN401
    """Vision rotary position embeddings driven by a *Python* grid list.

    Numerically identical to stock ``Qwen3VLVisionModel.rot_pos_emb``, but it
    iterates over the Python constant ``grid_thw_list`` instead of
    ``grid_thw.tolist()``. Under ``torch.export`` the baked ``grid_thw`` buffer is
    a lifted tensor input, so ``.tolist()`` would yield unbacked symints and the
    per-image ``arange`` / ``reshape`` shapes could not be resolved; the constant
    ints keep every shape concrete. For each image it enumerates the (row, col)
    patch coordinates in ``spatial_merge_size`` blocks, gathers their rotary
    frequencies from the shared table and flattens them.

    Args:
        visual: The vision tower (reads ``spatial_merge_size`` and ``rotary_pos_emb``).
        grid_thw_list: Per-image ``[t, h, w]`` geometry as plain Python ints.

    Returns:
        The flattened rotary position embeddings for all vision patches.
    """
    merge_size = visual.spatial_merge_size

    max_hw = max(max(h, w) for _, h, w in grid_thw_list)
    freq_table = visual.rotary_pos_emb(max_hw)
    device = freq_table.device

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw_list:
        # Patch coordinates are laid out in spatial_merge_size x spatial_merge_size
        # blocks (matching how the merger later folds neighbouring patches
        # together), so build row/col indices as block-offset + intra-block-offset.
        merged_h, merged_w = height // merge_size, width // merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            # Temporal frames share the same spatial grid, so repeat it.
            coords = coords.repeat(num_frames, 1)
        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]
    return embeddings.flatten(1)


def export_fast_pos_embed_interpolate(visual: Any, grid_thw_list: list[list[int]]) -> torch.Tensor:  # noqa: PLR0914, ANN401
    """Bilinearly interpolate the learned position embeddings to the grid.

    Numerically identical to stock ``Qwen3VLVisionModel.fast_pos_embed_interpolate``,
    but driven by the Python constant ``grid_thw_list`` so the per-image
    ``torch.linspace(0, num_grid_per_side - 1, h)`` calls take concrete sizes.
    Under ``torch.export`` those ``linspace`` bounds come from ``grid_thw`` (a
    lifted tensor input via ``.tolist()``), which triggers a data-dependent guard;
    the constant ints avoid it. For each image it maps the target ``h x w`` grid
    onto the learned ``num_grid_per_side`` grid, gathers the four surrounding
    embeddings and blends them with the fractional ``(dh, dw)`` bilinear weights,
    then reorders the patches into ``spatial_merge_size`` blocks.

    Args:
        visual: The vision tower (reads ``num_grid_per_side``, ``pos_embed`` and
            ``config.spatial_merge_size``).
        grid_thw_list: Per-image ``[t, h, w]`` geometry as plain Python ints.

    Returns:
        The interpolated position embeddings for all vision patches.
    """
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = visual.pos_embed.weight.device

    # Four accumulators = the four bilinear corners (floor/ceil x floor/ceil) of
    # the learned-grid neighbours for every target patch.
    idx_list: list[list[float]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]

    for _t, h, w in grid_thw_list:
        # Sample positions on the learned grid for the target h/w axes.
        h_idxs = torch.linspace(0, visual.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, visual.num_grid_per_side - 1, w)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=visual.num_grid_per_side - 1)
        # Fractional distances -> bilinear interpolation weights.
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        base_h = h_idxs_floor * visual.num_grid_per_side
        base_h_ceil = h_idxs_ceil * visual.num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=visual.pos_embed.weight.dtype, device=device)
    # Blend the four gathered corners with their bilinear weights.
    pos_embeds = visual.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    # Reorder each image's patches into spatial_merge_size blocks so they line up
    # with the tower's merged token ordering.
    patch_pos_embeds_permute = []
    merge_size = visual.config.spatial_merge_size
    for patch_pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = patch_pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    return torch.cat(patch_pos_embeds_permute)


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


def export_scatter_visual_embeds(
    inputs_embeds: torch.Tensor,
    image_token_indices: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """Merge visual embeds into token embeddings by integer index.

    Export-friendly replacement for the stock image/text merge, which uses
    ``masked_scatter`` (-> an unconvertible ``Where`` whose operand shapes
    disagree). The integer-index ``index_copy`` (-> ``ScatterND``) is numerically
    identical for a single-batch sequence.

    Args:
        inputs_embeds: ``(1, seq_len, hidden)`` token embeddings.
        image_token_indices: ``(num_visual,)`` positions of the image tokens.
        image_embeds: ``(num_visual, hidden)`` visual embeddings.

    Returns:
        ``(1, seq_len, hidden)`` embeddings with the image slots replaced.
    """
    merged = inputs_embeds[0].index_copy(0, image_token_indices, image_embeds)
    return merged.unsqueeze(0)


def export_add_deepstack_embeds(
    hidden_states: torch.Tensor,
    image_token_indices: torch.Tensor,
    visual_embeds: torch.Tensor,
) -> torch.Tensor:
    """Add deepstack visual features at the image-token positions by index.

    Export-friendly replacement for the stock ``_deepstack_process``, which adds
    ``visual_embeds`` into ``hidden_states`` via boolean-mask assignment (-> an
    unconvertible ``Where``). The ``index_select`` + ``index_copy`` variant
    (-> ``Gather`` / ``ScatterND``) is numerically identical for a single-batch
    sequence.

    Args:
        hidden_states: ``(1, seq_len, hidden)`` decoder hidden states.
        image_token_indices: ``(num_visual,)`` positions of the image tokens.
        visual_embeds: ``(num_visual, hidden)`` deepstack features to add.

    Returns:
        ``(1, seq_len, hidden)`` hidden states with the features added.
    """
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    row = hidden_states[0]
    updated = row.index_copy(0, image_token_indices, row.index_select(0, image_token_indices) + visual_embeds)
    return updated.unsqueeze(0)


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

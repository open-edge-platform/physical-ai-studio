# Copyright (C) 2026 Xiaomi Corporation.

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL backbone shim for the XR0 Vision-Language-Action policy.

The XR0 source ships a machine-generated verbatim copy of the stock
``transformers`` Qwen3-VL model (``xr0/mibot/models/VLM/qwen3vl.py``). The only
*functional* difference from the upstream model is that the copy surfaces the 3D
MRoPE ``position_ids`` (and the ``attention_mask``) on its output dataclasses --
``XR0.forward`` consumes ``vlm_outputs.position_ids.max(dim=-1)`` to continue the
MRoPE sequence into the DiT action head, plus ``vlm_outputs.past_key_values``.

Rather than vendor ~1500 lines of upstream model code (which is version-locked to
the transformers release it was generated from), this module subclasses the
installed stock :class:`~transformers.Qwen3VLForConditionalGeneration` and adds
back only that one behaviour: it computes the 3D position ids with the model's
own :meth:`compute_3d_position_ids` and attaches them to the returned output. All
VLM numerics are inherited unchanged from stock ``transformers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

from .export_openvino import (
    export_add_deepstack_embeds,
    export_build_additive_causal_mask,
    export_fast_pos_embed_interpolate,
    export_image_token_gather,
    export_mrope_position_ids,
    export_rot_pos_emb,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
)

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast


class XR0Qwen3VL(Qwen3VLForConditionalGeneration):
    """Stock Qwen3-VL that also exposes the 3D MRoPE ``position_ids``.

    Stock ``transformers`` computes the 3D position ids internally but discards
    them (only ``rope_deltas`` is returned). XR0's action head needs the full
    ``(3, batch, seq)`` grid, so this shim computes it up front via
    :meth:`~transformers.Qwen3VLModel.compute_3d_position_ids`, passes it into the
    stock forward (so the backbone uses exactly the exposed ids), and attaches it
    to the output as ``outputs.position_ids``.

    When ``mm_token_type_ids`` is not supplied (the Qwen3-VL processor normally
    provides it) it is derived from ``input_ids`` using the configured image and
    video token ids so the MRoPE index can still be built.
    """

    @torch.no_grad()
    def prepare_ingraph_export(self, image_grid_thw: torch.LongTensor) -> None:
        """Bake **only** the fixed image geometry as constants for the export.

        Args:
            image_grid_thw: The fixed vision geometry ``(num_images, 3)``.
        """
        merge = int(self.config.vision_config.spatial_merge_size)
        grid_list = [[int(dim) for dim in row] for row in image_grid_thw.tolist()]
        # Per image-token merged-grid offsets + per-block MRoPE advance. These
        # reproduce ``get_vision_position_ids`` / ``get_rope_index`` for the fixed
        # geometry: within a block token ``idx`` sits at ``(row, col) =
        # (idx // llm_w, idx % llm_w)`` and only the *last* token of each block
        # advances the running position (by ``max(llm_h, llm_w)``).
        rows: list[int] = []
        cols: list[int] = []
        advances: list[int] = []
        for grid_t, grid_h, grid_w in grid_list:
            llm_h = grid_h // merge
            llm_w = grid_w // merge
            block = [(idx // llm_w, idx % llm_w) for _ in range(grid_t) for idx in range(llm_h * llm_w)]
            for cursor, (row, col) in enumerate(block):
                rows.append(row)
                cols.append(col)
                advances.append(max(llm_h, llm_w) if cursor == len(block) - 1 else 0)
        device = image_grid_thw.device
        baked = {
            "_export_image_grid_thw": image_grid_thw.detach().clone(),
            "_export_image_row": torch.tensor(rows, dtype=torch.long, device=device),
            "_export_image_col": torch.tensor(cols, dtype=torch.long, device=device),
            "_export_image_advance": torch.tensor(advances, dtype=torch.long, device=device),
        }
        for name, tensor in baked.items():
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, tensor, persistent=False)
        # Keep the vision geometry as a *Python* constant too for the ``torch.export``:
        #
        # Torch.export lifts registered buffers as tensor inputs, so ``grid_thw.tolist()`` in
        # the vision tower would yield unbacked symints; the export-time
        # patch consumes these concrete ints instead (see :meth:`_ensure_export_patch`).
        self._export_grid_list = grid_list
        # Per-window token counts for the vision attention. Stock builds these
        # from ``cu_seqlens`` and calls ``lengths.tolist()``
        self._export_vision_seqlens = [h * w for t, h, w in grid_list for _ in range(t)]
        self._ingraph_export = True

    def _ensure_export_patch(self) -> None:
        """Swap the stock Qwen3-VL ops for their export-friendly equivalents.

        Installs the module-level ``export_*`` reimplementations onto the vision
        tower and language model.  Each is numerically identical to stock
        but OpenVINO-convertible;
        """
        if getattr(self, "_export_patched", False):
            return
        shim = self
        inner = self.model
        visual = inner.visual
        text_model = inner.language_model
        orig_model_forward = inner.forward
        orig_deepstack_process = text_model._deepstack_process  # noqa: SLF001

        def _make_vision_attn_forward(attn: torch.nn.Module) -> object:
            def _forward(
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor | None = None,  # noqa: ARG001
                rotary_pos_emb: torch.Tensor | None = None,  # noqa: ARG001
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs: object,  # noqa: ARG001
            ) -> torch.Tensor:
                return export_vision_attn_forward(
                    attn,
                    shim._export_vision_seqlens,  # noqa: SLF001
                    hidden_states,
                    cast("tuple[torch.Tensor, torch.Tensor]", position_embeddings),
                )

            return _forward

        def _patched_rot_pos_emb(grid_thw: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return export_rot_pos_emb(visual, shim._export_grid_list)  # noqa: SLF001

        def _patched_fast_pos_embed_interpolate(grid_thw: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return export_fast_pos_embed_interpolate(visual, shim._export_grid_list)  # noqa: SLF001

        def _patched_model_forward(
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: object | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
            **kwargs: object,
        ) -> Qwen3VLModelOutputWithPast:
            gather_index = getattr(shim, "_image_gather_index", None)
            image_mask = getattr(shim, "_image_token_mask", None)
            if gather_index is None or pixel_values is None or pixel_values_videos is not None:
                return orig_model_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    **kwargs,
                )
            if inputs_embeds is None:
                inputs_embeds = inner.get_input_embeddings()(input_ids)
            # Run the tower directly (not ``get_image_features``) so its
            # ``split_sizes.tolist()`` is skipped;
            # The two patched geometry helpers above already consume the
            # constant grid, and a freshly built *constant* grid tensor keeps the
            # tower's inline ``cu_seqlens`` (tensor ops on ``grid_thw``) concrete.
            grid_const = torch.tensor(
                shim._export_grid_list,  # noqa: SLF001
                dtype=torch.long,
                device=pixel_values.device,
            )
            # The graph's ``pixel_values`` input is already the flat patchified
            # ``pixel_values`` produced off-graph by the NumPy preprocessor
            # (temporal duplication + patchify reshape/transpose), so it is fed
            # straight to the vision tower with the baked constant grid.
            vision_output = visual(
                pixel_values.type(visual.dtype),
                grid_thw=grid_const,
                return_dict=True,
            )
            image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = vision_output.deepstack_features
            # OpenVINO-friendly merge: gather the image embeds to the sequence
            # length and blend by the ``{0, 1}`` image mask (``Gather`` + exact
            # ``0``/``1`` ``Mul``/``Add``), instead of ``masked_scatter`` (-> an
            # unconvertible ``Where``). Numerically identical for a single-batch
            # sequence.
            inputs_embeds = export_scatter_visual_embeds(inputs_embeds, gather_index, image_mask, image_embeds)
            visual_pos_masks = input_ids == inner.config.image_token_id
            if position_ids is None:
                position_ids = inner.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
            # Pre-build the 4-D additive causal mask so the text model's SDPA mask
            # builder early-exits instead of emitting a boolean ``GatherND`` the
            # OpenVINO GPU plugin cannot compile (see
            # :func:`export_build_additive_causal_mask`).
            if attention_mask is not None:
                attention_mask = export_build_additive_causal_mask(
                    attention_mask,
                    cast("torch.Tensor", inputs_embeds).dtype,
                )
            outputs = inner.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
                **kwargs,
            )
            return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=inner.rope_deltas)

        def _patched_deepstack_process(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            """Add the deepstack visual features at the image-token positions.

            Thin wrapper over :func:`export_add_deepstack_embeds` using the
            traceable image-token gather; falls back to stock when it is absent
            (normal, non-export inference).

            Returns:
                ``hidden_states`` with the deepstack features added.
            """
            gather_index = getattr(shim, "_image_gather_index", None)
            image_mask = getattr(shim, "_image_token_mask", None)
            if gather_index is None:
                return orig_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
            return export_add_deepstack_embeds(hidden_states, gather_index, image_mask, visual_embeds)

        inner.forward = _patched_model_forward
        text_model._deepstack_process = _patched_deepstack_process  # noqa: SLF001
        visual.rot_pos_emb = _patched_rot_pos_emb
        visual.fast_pos_embed_interpolate = _patched_fast_pos_embed_interpolate
        for block in visual.blocks:
            block.attn.forward = _make_vision_attn_forward(block.attn)
        self._export_patched = True

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,  # noqa: ANN001
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: object,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """Run the stock forward and attach the 3D MRoPE ``position_ids``.

        On the normal eager path this derives ``mm_token_type_ids`` /
        ``position_ids`` (when absent) and delegates to the stock forward,
        re-exposing the 3D grid on the output.

        When in-graph export mode is active (after
        :meth:`prepare_ingraph_export`), the fixed image geometry is taken from
        the baked constant buffers, the ``position_ids`` and the image-token
        gather are recomputed traceably for the runtime prompt via
        :func:`export_mrope_position_ids` / :func:`export_image_token_gather`, and
        the stock ops are swapped for their OpenVINO-convertible ``export_*``
        equivalents (see :meth:`_ensure_export_patch`). This keeps the traced
        graph free of the data-dependent Python control flow that ``torch.export``
        cannot capture.

        Returns:
            The stock Qwen3-VL output with the 3D MRoPE ``position_ids`` attached.
        """
        if getattr(self, "_ingraph_export", False):
            image_grid_thw = self._export_image_grid_thw
            gather_index, image_mask = export_image_token_gather(
                input_ids,
                self.config.image_token_id,
                self._export_image_row.shape[0],
            )
            position_ids = export_mrope_position_ids(
                attention_mask,
                gather_index,
                image_mask,
                self._export_image_row,
                self._export_image_col,
                self._export_image_advance,
            )
            self._image_gather_index = gather_index
            self._image_token_mask = image_mask
            self._ensure_export_patch()
            if mm_token_type_ids is None and input_ids is not None:
                # Image tokens -> 1 (XR0 has no video). A pure elementwise cast, so
                # it traces without the boolean-scatter ``Where`` of the masked
                # assignment used on the eager path below.
                mm_token_type_ids = cast(
                    "torch.IntTensor",
                    (input_ids == self.config.image_token_id).to(torch.int32),
                )
        elif (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            derived_ids = torch.zeros_like(input_ids)
            derived_ids[input_ids == self.config.image_token_id] = 1
            derived_ids[input_ids == self.config.video_token_id] = 2
            mm_token_type_ids = cast("torch.IntTensor", derived_ids)

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        outputs.position_ids = position_ids
        return outputs

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
    export_rot_pos_emb,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
    patchify_image_grid,
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

    def build_3d_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor:
        """Compute the 3D MRoPE ``position_ids`` for the given inputs.

        This wraps the stock :meth:`~transformers.Qwen3VLModel.compute_3d_position_ids`
        (deriving ``mm_token_type_ids`` from ``input_ids`` when absent). The
        underlying ``get_rope_index`` uses data-dependent Python control flow
        (``tensor.tolist()`` / :func:`itertools.groupby`), so this **must** run
        eagerly on concrete tensors -- it cannot be captured by ``torch.export``.

        Returns:
            The 3D MRoPE ``position_ids`` tensor (shape ``(3, batch, seq)``).
        """
        if (
            mm_token_type_ids is None
            and input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            derived_ids = torch.zeros_like(input_ids)
            derived_ids[input_ids == self.config.image_token_id] = 1
            derived_ids[input_ids == self.config.video_token_id] = 2
            mm_token_type_ids = cast("torch.IntTensor", derived_ids)

        return self.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=None,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )

    @torch.no_grad()
    def prepare_ingraph_export(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> None:
        """Bake the fixed image geometry as constants for a self-contained export.

        For a fixed image size the ``image_grid_thw`` geometry, the image-token
        positions and the MRoPE ``position_ids`` of the fixed prefix (system
        prompt + image grid) are deterministic. This precomputes them once from a
        representative (right-padded) sample and stores them as non-persistent
        constant buffers, then enables in-graph export mode.

        Args:
            input_ids: Token ids of the representative padded prompt ``(1, L)``.
            attention_mask: Attention mask of the same prompt ``(1, L)``.
            image_grid_thw: The fixed vision geometry ``(num_images, 3)``.
        """
        position_ids = self.build_3d_position_ids(
            input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )
        image_token_indices = self.image_token_positions(input_ids)
        for name, tensor in (
            ("_export_image_grid_thw", image_grid_thw),
            ("_export_position_ids", position_ids),
            ("_export_image_token_indices", image_token_indices),
        ):
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, tensor.detach().clone(), persistent=False)
        # The fixed system-prompt text and the fixed image grid
        # (Everything up to and including the last image token)
        # has deterministic MRoPE  positions, so it stays baked.
        #
        # The *task text* after the image varies in  length between prompts,
        # so its positions must be recomputed at inference from the runtime ``attention_mask``;
        #
        # The post-image text is plain 1D sequential (all three MRoPE axes equal), so
        # we only need where it starts and its first position value.
        post_image_start = int(image_token_indices.max().item()) + 1
        self._export_post_image_start = post_image_start
        self._export_post_image_base = int(position_ids[0, 0, post_image_start].item())
        # Keep the vision geometry as a *Python* constant too for the ``torch.export``:
        #
        # Torch.export lifts registered buffers as tensor inputs, so ``grid_thw.tolist()`` in
        # the vision tower would yield unbacked symints; the export-time
        # patch consumes these concrete ints instead (see :meth:`_ensure_export_patch`).
        self._export_grid_list = [[int(dim) for dim in row] for row in image_grid_thw.tolist()]
        # Per-window token counts for the vision attention. Stock builds these
        # from ``cu_seqlens`` and calls ``lengths.tolist()``
        self._export_vision_seqlens = [h * w for t, h, w in self._export_grid_list for _ in range(t)]
        self._ingraph_export = True

    def _runtime_export_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Rebuild the export MRoPE ``position_ids`` for the runtime prompt length.

        Args:
            attention_mask: The runtime attention mask ``(1, L)``.

        Returns:
            The 3D MRoPE ``position_ids`` tensor ``(3, 1, L)`` for this prompt.
        """
        baked = self._export_position_ids
        seq_len = baked.shape[-1]
        mask = attention_mask.reshape(-1)[:seq_len].to(torch.long)
        seq_index = torch.arange(seq_len, device=baked.device)
        in_tail = seq_index >= self._export_post_image_start
        valid_tail = mask * in_tail.to(torch.long)
        # 0-based running index among the valid post-image tokens.
        tail_index = torch.cumsum(valid_tail, dim=0) - 1
        tail_pos = (self._export_post_image_base + tail_index).clamp_min(0)
        use_tail = (in_tail & (mask > 0)).reshape(1, 1, seq_len).expand_as(baked)
        tail_pos = tail_pos.reshape(1, 1, seq_len).expand_as(baked)
        return torch.where(use_tail, tail_pos, baked)

    def image_token_positions(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return the integer sequence positions of the image tokens.

        Returns:
            A ``(num_visual_tokens,)`` long tensor of image-token positions in the
            (single-batch) sequence.
        """
        return (input_ids[0] == self.config.image_token_id).nonzero(as_tuple=True)[0]

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
            idx = shim._image_token_indices  # noqa: SLF001
            if idx is None or pixel_values is None or pixel_values_videos is not None:
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
            # The graph's ``pixel_values`` input is the pre-patchify normalized image
            # grid ``(num_images, C, H, W)``; bake the Qwen3-VL temporal-duplication +
            # patchify reshape/transpose into the graph (constant geometry from the
            # baked grid) so the Runtime preprocessor does not have to reproduce it.
            pixel_values = patchify_image_grid(
                pixel_values,
                shim._export_grid_list,  # noqa: SLF001
                temporal_patch_size=visual.config.temporal_patch_size,
                patch_size=visual.config.patch_size,
                merge_size=visual.config.spatial_merge_size,
            )
            vision_output = visual(
                pixel_values.type(visual.dtype),
                grid_thw=grid_const,
                return_dict=True,
            )
            image_embeds = vision_output.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = vision_output.deepstack_features
            # OpenVINO-friendly merge: integer-index scatter (``index_copy`` ->
            # ``ScatterND``) instead of ``masked_scatter`` (-> unconvertible
            # ``Where``). Numerically identical for a single-batch sequence.
            inputs_embeds = export_scatter_visual_embeds(inputs_embeds, idx, image_embeds)
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

            Thin wrapper over :func:`export_add_deepstack_embeds` using the baked
            image-token indices; falls back to stock when they are absent (normal,
            non-export inference).

            Returns:
                ``hidden_states`` with the deepstack features added.
            """
            idx = shim._image_token_indices  # noqa: SLF001
            if idx is None:
                return orig_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)
            return export_add_deepstack_embeds(hidden_states, idx, visual_embeds)

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
        :meth:`prepare_ingraph_export`), the fixed image geometry and the prefix
        MRoPE positions are taken from the baked constant buffers, the
        ``position_ids`` are rebuilt for the runtime prompt length via
        :meth:`_runtime_export_position_ids`, and the stock ops are swapped for
        their OpenVINO-convertible ``export_*`` equivalents (see
        :meth:`_ensure_export_patch`). This keeps the traced graph free of the
        data-dependent Python control flow that ``torch.export`` cannot capture.

        Returns:
            The stock Qwen3-VL output with the 3D MRoPE ``position_ids`` attached.
        """
        if getattr(self, "_ingraph_export", False):
            image_grid_thw = self._export_image_grid_thw
            position_ids = self._runtime_export_position_ids(attention_mask)
            self._image_token_indices = self._export_image_token_indices
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

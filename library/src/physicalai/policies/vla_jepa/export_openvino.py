# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA's single-observation OpenVINO graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from physicalai.inference.manifest import ComponentSpec
from torch import Tensor, nn
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from physicalai.data import FeatureType
from physicalai.policies.xr0.export_openvino import (
    export_add_deepstack_embeds,
    export_build_additive_causal_mask,
    export_image_token_gather,
    export_mrope_position_ids,
    export_precompute_vision_geometry,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
)

from .processors import VLAJEPAPostprocessor, VLAJEPAPreprocessor, _batch_key, _normalizer

if TYPE_CHECKING:
    from .config import VLAJEPAConfig
    from .model import VLAJEPAModel


class VLAJEPAOpenVINOModel(nn.Module):
    """Trace Qwen and the complete flow sampler without training-only world modules.

    State normalization and action denormalization run in-graph, so Runtime only
    tokenizes and patchifies.
    """

    interp_indices: Tensor
    interp_weights: Tensor
    position_ids: Tensor
    image_row: Tensor
    image_col: Tensor
    image_advance: Tensor

    def __init__(self, model: VLAJEPAModel, config: VLAJEPAConfig, image_grid_thw: Tensor) -> None:
        """Share inference weights and bake only the fixed camera geometry."""
        super().__init__()
        self.visual = model.qwen.visual
        self.text = model.qwen.language_model
        self.action_model = model.action_model
        device = next(model.parameters()).device
        self.normalizer = _normalizer(config).to(device)
        self.state_key = next((_batch_key(f) for f in config.input_features if f.ftype == FeatureType.STATE), None)
        self.postprocessor = VLAJEPAPostprocessor(config).to(device)
        self.image_token_id = model.qwen.config.image_token_id
        geometry = export_precompute_vision_geometry(self.visual, image_grid_thw)
        for name, tensor in geometry.items():
            self.register_buffer(name, tensor)
        merge = self.visual.spatial_merge_size
        rows, cols, advances = [], [], []
        self.split_sizes = []
        for temporal, height, width in image_grid_thw.tolist():
            self.split_sizes.extend([height * width] * temporal)
            merged_h, merged_w = height // merge, width // merge
            count = temporal * merged_h * merged_w
            rows.extend([index // merged_w for _ in range(temporal) for index in range(merged_h * merged_w)])
            cols.extend([index % merged_w for _ in range(temporal) for index in range(merged_h * merged_w)])
            advances.extend([0] * (count - 1) + [max(merged_h, merged_w)])
        for name, values in (("image_row", rows), ("image_col", cols), ("image_advance", advances)):
            self.register_buffer(name, torch.tensor(values, device=image_grid_thw.device, dtype=torch.long))

    def _vision(self, pixels: Tensor) -> tuple[Tensor, list[Tensor]]:
        visual = self.visual
        hidden = visual.patch_embed(pixels.to(visual.dtype))
        positions = (visual.pos_embed(self.interp_indices) * self.interp_weights[:, :, None]).sum(1)
        hidden = torch.add(hidden, positions.to(hidden.dtype))
        rotary = visual.rotary_pos_emb(self.position_ids).reshape(hidden.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        embeddings = (rotary.cos(), rotary.sin())
        deepstack = []
        for index, block in enumerate(visual.blocks):
            hidden = torch.add(
                hidden,
                export_vision_attn_forward(block.attn, self.split_sizes, block.norm1(hidden), embeddings),
            )
            hidden = torch.add(hidden, block.mlp(block.norm2(hidden)))
            if index in visual.deepstack_visual_indexes:
                deepstack.append(visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(index)](hidden))
        return visual.merger(hidden), deepstack

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Return denormalized actions, preserving the final pre-RMSNorm readout."""
        image_embeds, deepstack = self._vision(batch["pixel_values"])
        gather, mask = export_image_token_gather(batch["input_ids"], self.image_token_id, self.image_row.shape[0])
        hidden = export_scatter_visual_embeds(self.text.embed_tokens(batch["input_ids"]), gather, mask, image_embeds)
        positions = export_mrope_position_ids(
            batch["attention_mask"],
            gather,
            mask,
            self.image_row,
            self.image_col,
            self.image_advance,
        )
        attention = export_build_additive_causal_mask(batch["attention_mask"], hidden.dtype)
        embeddings = self.text.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.text.layers):
            hidden = layer(hidden, attention_mask=attention, position_embeddings=embeddings, use_cache=False)
            if index < len(deepstack):
                hidden = export_add_deepstack_embeds(hidden, gather, mask, deepstack[index])
        indices = batch["embodied_positions"][:, :, None].expand(-1, -1, hidden.shape[-1])
        dtype = next(self.action_model.parameters()).dtype
        context = {"tokens": hidden.gather(1, indices).to(dtype)}
        if self.state_key is not None:
            state = self.normalizer({self.state_key: batch["state"].float()})[self.state_key]
            context["state"] = state[:, None].to(dtype)
        actions = self.action_model.sample(context, noise=batch["noise"].to(dtype)).float()
        return self.postprocessor(actions)


def runtime_preprocessor_spec(preprocessor: VLAJEPAPreprocessor) -> ComponentSpec:
    """Bake the Runtime ``vla_jepa`` preprocessor: camera geometry, chat prompt and token ids.

    Returns:
        A manifest component that needs only Runtime and the exported ``tokenizer.json``.
    """
    config = preprocessor.config
    image = preprocessor.processor.image_processor
    patch, merge = image.patch_size, image.merge_size
    height, width = smart_resize(
        *config.image_resolution,
        factor=patch * merge,
        min_pixels=image.size["shortest_edge"],
        max_pixels=image.size["longest_edge"],
    )
    text = config.prompt_template.format(
        instruction="{instruction}",
        actions=preprocessor.latent_prompt,
        e_actions=preprocessor.embodied_prompt,
    )
    content = [{"type": "image"} for _ in preprocessor.image_keys] + [{"type": "text", "text": text}]
    prompt = preprocessor.processor.apply_chat_template(
        [[{"role": "user", "content": content}]],
        tokenize=False,
        add_generation_prompt=True,
    )[0]
    pad = preprocessor.processor.image_token
    return ComponentSpec(
        type="vla_jepa",
        artifact="tokenizer.json",
        prompt=prompt.replace(pad, pad * ((height // patch) * (width // patch) // merge**2)),
        image_keys=preprocessor.image_keys,
        embodied_token_id=config.embodied_action_token_id,
        num_embodied_tokens=config.num_embodied_action_tokens,
        noise_shape=[config.chunk_size, config.action_dim],
        image_resolution=list(config.image_resolution),
        vision_resolution=[height, width],
        image_mean=list(image.image_mean),
        image_std=list(image.image_std),
        patch_size=patch,
        merge_size=merge,
        temporal_patch_size=image.temporal_patch_size,
    )

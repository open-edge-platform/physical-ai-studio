# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA's single-observation OpenVINO graph and offline Runtime inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from physicalai.inference.preprocessors.base import Preprocessor
from torch import Tensor, nn

from physicalai.policies.xr0.export_openvino import (
    export_add_deepstack_embeds,
    export_build_additive_causal_mask,
    export_image_token_gather,
    export_mrope_position_ids,
    export_precompute_vision_geometry,
    export_scatter_visual_embeds,
    export_vision_attn_forward,
)

from .config import VLAJEPAConfig
from .processors import VLAJEPAPostprocessor, VLAJEPAPreprocessor

if TYPE_CHECKING:
    from .model import VLAJEPAModel

_GRID_DIMENSIONS = 3


class VLAJEPAOpenVINOPreprocessor(Preprocessor):
    """Reuse the training processor locally; no model weights or Hub access needed."""

    def __init__(self, artifact: str) -> None:
        """Restore feature statistics, prompts, and tokenizer from a local JSON artifact.

        Raises:
            TypeError: If the artifact configuration or camera grids are malformed.
        """
        with Path(artifact).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
            msg = "VLA-JEPA processor artifact must contain a configuration object."
            raise TypeError(msg)
        grid = data.get("image_grid_thw")
        if (
            not isinstance(grid, list)
            or not grid
            or any(
                not isinstance(row, list)
                or len(row) != _GRID_DIMENSIONS
                or any(type(dim) is not int or dim <= 0 for dim in row)
                for row in grid
            )
        ):
            msg = "VLA-JEPA processor artifact requires positive integer image grids."
            raise TypeError(msg)
        self.config = VLAJEPAConfig.from_dict(data["config"])
        self.processor = VLAJEPAPreprocessor(self.config)
        self.image_grid_thw = torch.tensor(grid, dtype=torch.long)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return tensor inputs and fresh flow noise for one observation.

        Raises:
            ValueError: If batch size, image geometry, or noise differs from the graph contract.
        """
        # Runtime flattens nested observations only after preprocessing, e.g. {"images": {"front": ...}}.
        inputs = {
            **inputs,
            **{
                f"{name}.{key}": item
                for name, value in inputs.items()
                if isinstance(value, dict)
                for key, item in value.items()
            },
        }
        keys = {*self.processor.image_keys, *self.processor.state_keys, "images", "task"}
        batch = {
            name: (value.tolist() if name == "task" else torch.from_numpy(value))
            if isinstance(value, np.ndarray)
            else value
            for name, value in inputs.items()
            if name in keys
        }
        processed = self.processor(batch)
        if processed["input_ids"].shape[0] != 1:
            msg = "VLA-JEPA OpenVINO inference requires batch size one."
            raise ValueError(msg)
        if not torch.equal(processed["image_grid_thw"].cpu(), self.image_grid_thw):
            msg = "Image geometry differs from the exported VLA-JEPA graph."
            raise ValueError(msg)
        shape = (1, self.config.chunk_size, self.config.action_dim)
        noise = inputs.get("noise")
        noise = torch.randn(shape) if noise is None else torch.as_tensor(noise, dtype=torch.float32)
        if tuple(noise.shape) != shape or not torch.isfinite(noise).all():
            msg = f"Flow noise must be finite and have shape {shape}."
            raise ValueError(msg)
        result = {
            name: processed[name] for name in ("input_ids", "attention_mask", "pixel_values", "embodied_positions")
        }
        if self.processor.state_keys:
            result["state"] = processed["state"]
        result["noise"] = noise
        return {name: tensor.detach().cpu().numpy() for name, tensor in result.items()}


class VLAJEPAOpenVINOModel(nn.Module):
    """Trace Qwen and the complete flow sampler without training-only world modules."""

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
        self.postprocessor = VLAJEPAPostprocessor(config).to(next(model.parameters()).device)
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
        if "state" in batch:
            context["state"] = batch["state"].to(dtype)
        actions = self.action_model.sample(context, noise=batch["noise"].to(dtype)).float()
        return self.postprocessor(actions)

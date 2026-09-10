# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team.
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Assemble backbone-ready MolmoAct2 model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from physicalai.data.constants import IMAGES, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK

if TYPE_CHECKING:
    from .image import MolmoAct2ImageProcessor


_PACKED_IMAGE_DIM = 5


@dataclass(frozen=True)
class MolmoAct2InputLayout:
    """Describe model-input dimensions and image token layout.

    Steps:
        1. Store environment and fixed model action dimensions.
        2. Store image placeholder, boundary, patch, and column tokens.
        3. Expose configured image tokens for token-type classification.
    """

    env_action_dim: int
    max_action_dim: int
    image_placeholder_token_id: int
    image_patch_id: int
    image_start_token_id: int
    image_end_token_id: int
    image_col_id: int | None = None
    low_res_image_start_token_id: int | None = None
    frame_start_token_id: int | None = None
    frame_end_token_id: int | None = None
    image_low_res_id: int | None = None
    image_use_col_tokens: bool = True
    use_single_crop_col_tokens: bool | None = False
    use_single_crop_start_token: bool = True

    @property
    def image_token_ids(self) -> list[int]:
        """All configured token IDs that identify image content."""
        values = (
            self.image_patch_id,
            self.image_col_id,
            self.image_start_token_id,
            self.low_res_image_start_token_id,
            self.frame_start_token_id,
            self.image_end_token_id,
            self.frame_end_token_id,
            self.image_low_res_id,
        )
        return [int(value) for value in values if value is not None]


def _default_action_dim_is_pad(layout: MolmoAct2InputLayout, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((batch_size, layout.max_action_dim), dtype=torch.bool, device=device)
    mask[:, : layout.env_action_dim] = False
    return mask


def _build_token_type_ids(layout: MolmoAct2InputLayout, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_set = torch.tensor(layout.image_token_ids, device=ids.device, dtype=ids.dtype)
    return (ids.unsqueeze(-1) == token_set.view(1, 1, -1)).any(-1).long() * mask.long()


def _image_token_ids_for_grid(layout: MolmoAct2InputLayout, grid: torch.Tensor) -> list[int]:
    """Expand one image grid into model image-token IDs.

    Returns:
        list[int]: A list of image-token IDs corresponding to the expanded grid.
    """
    resized_h, resized_w, height, width = (int(value) for value in grid.tolist())

    def rows(row_count: int, column_count: int, *, use_col: bool) -> list[int]:
        row = [layout.image_patch_id] * column_count
        if use_col and layout.image_col_id is not None:
            row.append(layout.image_col_id)
        return row * row_count

    single_col = (
        layout.image_use_col_tokens if layout.use_single_crop_col_tokens is None else layout.use_single_crop_col_tokens
    )
    if height == 0 or width == 0:
        return [
            layout.image_start_token_id,
            *rows(resized_h, resized_w, use_col=single_col),
            layout.image_end_token_id,
        ]
    low_start = (
        layout.low_res_image_start_token_id
        if layout.use_single_crop_start_token and layout.low_res_image_start_token_id is not None
        else layout.image_start_token_id
    )
    return [
        low_start,
        *rows(resized_h, resized_w, use_col=single_col),
        layout.image_end_token_id,
        layout.image_start_token_id,
        *rows(height, width, use_col=layout.image_use_col_tokens),
        layout.image_end_token_id,
    ]


def _expand_image_placeholders(
    *,
    layout: MolmoAct2InputLayout,
    pad_token_id: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace image placeholders with expanded image-token grids.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of input IDs, attention mask and token IDs.

    Raises:
        ValueError: If there are not enough image grids to expand all image placeholders.
    """
    rows: list[list[int]] = []
    widths: list[int] = []
    grid_index = 0
    for batch_index in range(input_ids.shape[0]):
        valid = attention_mask[batch_index].bool()
        expanded: list[int] = []
        for token in input_ids[batch_index][valid].tolist():
            if int(token) == layout.image_placeholder_token_id:
                if grid_index >= image_grids.shape[0]:
                    msg = "Not enough image grids to expand all image placeholders."
                    raise ValueError(msg)
                expanded.extend(_image_token_ids_for_grid(layout, image_grids[grid_index]))
                grid_index += 1
            else:
                expanded.append(int(token))
        rows.append(expanded)
        widths.append(len(expanded) + int((~valid).sum()))
    width = max(widths, default=1)
    output_ids = torch.full((len(rows), width), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    output_mask = torch.zeros((len(rows), width), dtype=attention_mask.dtype, device=attention_mask.device)
    for index, row in enumerate(rows):
        output_ids[index, : len(row)] = torch.tensor(row, dtype=input_ids.dtype, device=input_ids.device)
        output_mask[index, : len(row)] = 1
    return output_ids, output_mask, _build_token_type_ids(layout, output_ids, output_mask)


def _rebase_pooling_block(
    block: torch.Tensor,
    pooled_counts: torch.Tensor,
    patch_counts: torch.Tensor,
) -> torch.Tensor:
    """Rebase image-local patch indices within one example's pooling block.

    Returns:
        Pooling indices rebased into the example-level flattened patch tensor.
    """
    row = 0
    patch_offset = 0
    for item_count_tensor, patch_count_tensor in zip(pooled_counts, patch_counts, strict=True):
        item_count = int(item_count_tensor)
        item = block[row : row + item_count]
        block[row : row + item_count] = torch.where(item >= 0, item + patch_offset, item)
        patch_offset += int(patch_count_tensor)
        row += item_count
    return block


def _build_batched_images(  # noqa: PLR0914
    layout: MolmoAct2InputLayout,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_token_pooling: torch.Tensor,
    image_grids: torch.Tensor,
    image_num_crops: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regroup concatenated image crops and pooling per batch example.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the regrouped images and pooling tensors.

    Raises:
        ValueError: If image counts cannot be inferred from image-end tokens.
    """
    raw_counts = (input_ids == layout.image_end_token_id).sum(1)
    total_images = int(image_grids.shape[0])
    total_end_tokens = int(raw_counts.sum())
    if total_images == 0:
        counts = torch.zeros_like(raw_counts)
    elif total_end_tokens == total_images:
        counts = raw_counts
    elif total_end_tokens == 2 * total_images:
        counts = raw_counts // 2
    else:
        msg = (
            "Could not infer image counts from image-end tokens: "
            f"end_tokens={total_end_tokens}, image_grids={total_images}."
        )
        raise ValueError(msg)
    pooled_per_image = (image_grids[:, :2].prod(1) + image_grids[:, 2:].prod(1)).to(image_num_crops.dtype)
    example_for_image = torch.arange(counts.shape[0], device=input_ids.device).repeat_interleave(counts)
    crops_per_example = torch.zeros(counts.shape[0], dtype=image_num_crops.dtype, device=input_ids.device)
    crops_per_example.index_add_(0, example_for_image, image_num_crops)
    pooled_per_example = torch.zeros(counts.shape[0], dtype=pooled_per_image.dtype, device=input_ids.device)
    pooled_per_example.index_add_(0, example_for_image, pooled_per_image)
    images = torch.full(
        (counts.shape[0], int(crops_per_example.max()), pixel_values.shape[1], pixel_values.shape[2]),
        -1.0,
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )
    pooling = torch.full(
        (counts.shape[0], int(pooled_per_example.max()), image_token_pooling.shape[-1]),
        -1,
        dtype=image_token_pooling.dtype,
        device=image_token_pooling.device,
    )
    crop_offset = pooled_offset = image_offset = 0
    for example in range(counts.shape[0]):
        crop_count = int(crops_per_example[example])
        pooled_count = int(pooled_per_example[example])
        images[example, :crop_count] = pixel_values[crop_offset : crop_offset + crop_count]
        block = image_token_pooling[pooled_offset : pooled_offset + pooled_count].clone()
        image_count = int(counts[example])
        pooling[example, :pooled_count] = _rebase_pooling_block(
            block,
            pooled_per_image[image_offset : image_offset + image_count],
            (image_num_crops * pixel_values.shape[1])[image_offset : image_offset + image_count],
        )
        crop_offset += crop_count
        pooled_offset += pooled_count
        image_offset += image_count
    return images, pooling


def build_model_inputs(
    batch: dict[str, torch.Tensor],
    *,
    layout: MolmoAct2InputLayout,
    image_processor: MolmoAct2ImageProcessor,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Assemble backbone-ready tensors from packed preprocessing outputs.

    Returns:
        Text, vision, pooling, and action-mask tensors expected by the model.

    Raises:
        ValueError: If packed image dimensions or image metadata are inconsistent.
    """
    input_ids = batch[TOKENIZED_PROMPT]
    attention_mask = batch.get(TOKENIZED_PROMPT_MASK, torch.ones_like(input_ids))
    images = batch[IMAGES]
    if images.ndim != _PACKED_IMAGE_DIM:
        msg = f"Expected packed images [N, B, C, H, W], got {tuple(images.shape)}."
        raise ValueError(msg)
    num_images, batch_size, channels, height, width = images.shape
    flat_images = images.permute(1, 0, 2, 3, 4).reshape(batch_size * num_images, channels, height, width)
    image_output = image_processor(flat_images)
    input_ids, attention_mask, token_type_ids = _expand_image_placeholders(
        layout=layout,
        pad_token_id=pad_token_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grids=image_output["image_grids"].to(input_ids.device),
    )
    batched_images, pooling = _build_batched_images(
        layout,
        input_ids,
        image_output["pixel_values"].to(input_ids.device),
        image_output["image_token_pooling"].to(input_ids.device),
        image_output["image_grids"].to(input_ids.device),
        image_output["image_num_crops"].to(input_ids.device),
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "images": batched_images,
        "token_pooling": pooling,
        "action_dim_is_pad": _default_action_dim_is_pad(layout, batch_size, input_ids.device),
    }

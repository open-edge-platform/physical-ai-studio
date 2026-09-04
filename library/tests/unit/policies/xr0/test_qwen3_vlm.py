# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XR0 Qwen3-VL backbone shim (``qwen3_vlm``).

Fast, self-contained tests on a tiny synthetic Qwen3-VL (no HuggingFace
downloads). They cover the two behaviours the shim adds on top of stock
``transformers``: surfacing the 3D MRoPE ``position_ids`` (pinned against
weight-independent reference index math) and the in-graph export op swaps
(pinned to leave eager logits unchanged). Stock VLM numerics are inherited and
not re-tested.
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from physicalai.policies.xr0.export_openvino import patchify_image_grid
from physicalai.policies.xr0.qwen3_vlm import XR0Qwen3VL

# Special token ids for the tiny vocabulary.
IMAGE_TOKEN_ID = 151
VIDEO_TOKEN_ID = 152
VISION_START_TOKEN_ID = 150

# One image: grid (t, h, w) -> (t*h*w) / merge**2 merged image tokens.
IMAGE_GRID = (2, 4, 4)
SPATIAL_MERGE = 2
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
N_IMAGE_TOKENS = (IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2]) // SPATIAL_MERGE**2

# In-graph export parity uses a still image (grid_t == 1) so ``patchify_image_grid``
# can reproduce the exact flat ``pixel_values`` the vision tower consumes.
EXPORT_GRID = (1, 4, 4)
N_EXPORT_TOKENS = (EXPORT_GRID[0] * EXPORT_GRID[1] * EXPORT_GRID[2]) // SPATIAL_MERGE**2


def _config() -> Qwen3VLConfig:
    """Build a tiny Qwen3-VL config that supports a real multimodal forward."""
    vision = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        depth=2,
        out_hidden_size=64,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    text = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=200,
        rope_scaling={"type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": False},
    )
    return Qwen3VLConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
    )


# Reference 3D MRoPE position ids for the batch below. They depend only on the
# token layout and image grid (pure index math, weight-independent), so they are
# pinned exactly. Text tokens advance all three axes together; the image span
# holds the temporal/height/width grid before text resumes.
REFERENCE_POSITION_IDS = torch.tensor(
    [
        [[0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 5, 6, 7]],
        [[0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7]],
        [[0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 7]],
    ]
)


def _batch() -> dict:
    """Build a deterministic single-image multimodal batch."""
    grid = torch.tensor([list(IMAGE_GRID)])
    num_patches = int(grid.prod(-1).item())
    patch_dim = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size**2
    torch.manual_seed(0)
    pixel_values = torch.randn(num_patches, patch_dim)
    input_ids = torch.tensor([[5, 6, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * N_IMAGE_TOKENS), 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
    }


def _build_shim() -> XR0Qwen3VL:
    """Build the shim on a tiny synthetic Qwen3-VL config."""
    torch.manual_seed(0)
    return XR0Qwen3VL(_config()).eval()


class TestPositionIds:
    """The shim computes and surfaces the 3D MRoPE position ids stock discards."""

    def test_position_ids_reference(self) -> None:
        shim = _build_shim()
        with torch.no_grad():
            out = shim(**_batch(), use_cache=True)
        assert out.position_ids.shape == (3, 1, REFERENCE_POSITION_IDS.shape[-1])
        assert torch.equal(out.position_ids, REFERENCE_POSITION_IDS)

    def test_derived_token_types_match_reference(self) -> None:
        # The shim derives ``mm_token_type_ids`` from ``input_ids`` when absent.
        # Whether derived or passed explicitly, the position ids must match the
        # reference, proving the derivation reproduces the processor's labels.
        shim = _build_shim()
        batch = _batch()
        explicit = torch.zeros_like(batch["input_ids"])
        explicit[batch["input_ids"] == IMAGE_TOKEN_ID] = 1
        with torch.no_grad():
            out_derived = shim(**batch, use_cache=True)
            out_explicit = shim(**batch, mm_token_type_ids=explicit, use_cache=True)
        assert torch.equal(out_derived.position_ids, REFERENCE_POSITION_IDS)
        assert torch.equal(out_explicit.position_ids, REFERENCE_POSITION_IDS)

    def test_explicit_position_ids_passthrough(self) -> None:
        # When position ids are supplied the shim returns them untouched instead
        # of recomputing them.
        shim = _build_shim()
        batch = _batch()
        seq_len = batch["input_ids"].shape[1]
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)
        with torch.no_grad():
            out = shim(**batch, position_ids=position_ids, use_cache=True)
        assert torch.equal(out.position_ids, position_ids)


def _export_batch() -> tuple[dict, torch.Tensor]:
    """Build a still-image batch, returning ``(eager_batch, raw_image_grid)``.

    The eager batch carries the flat patchified ``pixel_values`` the tower
    consumes; the raw ``(num_images, C, H, W)`` grid is what the in-graph export
    forward patchifies internally, so both paths see the identical image.
    """
    grid = torch.tensor([list(EXPORT_GRID)])
    height = EXPORT_GRID[1] * PATCH_SIZE
    width = EXPORT_GRID[2] * PATCH_SIZE
    torch.manual_seed(0)
    raw_image = torch.randn(1, 3, height, width)
    pixel_values = patchify_image_grid(
        raw_image,
        [list(EXPORT_GRID)],
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        merge_size=SPATIAL_MERGE,
    )
    input_ids = torch.tensor([[5, 6, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * N_EXPORT_TOKENS), 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid,
    }
    return batch, raw_image


class TestIngraphExportParity:
    """``prepare_ingraph_export`` swaps stock ops for export-friendly equivalents.

    Each swapped op is numerically identical to stock, so enabling in-graph
    export mode must not materially change the logits for the same inputs.
    """

    def test_export_logits_match_eager(self) -> None:
        shim = _build_shim()
        batch, raw_image = _export_batch()
        with torch.no_grad():
            eager = shim(**batch, use_cache=True)
        shim.prepare_ingraph_export(
            batch["input_ids"],
            batch["attention_mask"],
            batch["image_grid_thw"],
        )
        with torch.no_grad():
            exported = shim(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=raw_image,
                use_cache=True,
            )
        assert exported.logits.shape == eager.logits.shape
        assert torch.allclose(exported.logits, eager.logits, atol=1e-3, rtol=1e-3)

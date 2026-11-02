# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parity test: graph-safe Qwen3-VL vision encoder vs the eager vision model.

Builds a tiny ``Qwen3VLVisionModel`` (random weights), then checks that the
static :class:`GraphSafeQwen3VLVisionModel` reproduces the eager forward output
bit-for-bit for a fixed ``grid_thw`` -- and that ``restore()`` undoes the
in-place attention swap.
"""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.rldx1.components.backbone.graph_safe_vision import (
    GraphSafeQwen3VLVisionAttention,
    GraphSafeQwen3VLVisionModel,
)


@pytest.fixture
def tiny_vision_model():  # noqa: ANN201
    """Build a tiny eager Qwen3-VL vision model, or skip if unavailable."""
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

        from physicalai.policies.rldx1.components.backbone.modeling_qwen3_vl import Qwen3VLVisionModel
    except Exception as exc:  # noqa: BLE001 - version drift / import failure -> skip
        pytest.skip(f"Qwen3VL vision model unavailable: {exc}")

    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0],
    )
    config._attn_implementation = "sdpa"  # noqa: SLF001
    try:
        model = Qwen3VLVisionModel(config).eval()
    except Exception as exc:  # noqa: BLE001 - config drift -> skip
        pytest.skip(f"Could not build tiny Qwen3VLVisionModel: {exc}")
    return model, config


def _make_pixel_values(config, grid_thw: torch.Tensor) -> torch.Tensor:
    """Random patch features matching the vision patch-embed input dim."""
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    num_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    return torch.randn(num_patches, patch_dim)


def test_graph_safe_vision_matches_eager(tiny_vision_model) -> None:  # noqa: ANN001
    """Static graph-safe vision forward reproduces the eager forward."""
    model, config = tiny_vision_model
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    pixel_values = _make_pixel_values(config, grid_thw)

    with torch.no_grad():
        eager_feat, eager_deepstack = model(pixel_values, grid_thw)

        gs = GraphSafeQwen3VLVisionModel(model, grid_thw)
        gs_feat, gs_deepstack = gs(pixel_values)

    torch.testing.assert_close(gs_feat, eager_feat, atol=1e-5, rtol=1e-4)
    assert len(gs_deepstack) == len(eager_deepstack)
    for gs_ds, eager_ds in zip(gs_deepstack, eager_deepstack, strict=True):
        torch.testing.assert_close(gs_ds, eager_ds, atol=1e-5, rtol=1e-4)


def test_graph_safe_vision_restore(tiny_vision_model) -> None:  # noqa: ANN001
    """restore() reinstates the original attention modules on every block."""
    model, _config = tiny_vision_model
    original_attns = [blk.attn for blk in model.blocks]
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    gs = GraphSafeQwen3VLVisionModel(model, grid_thw)
    assert all(isinstance(blk.attn, GraphSafeQwen3VLVisionAttention) for blk in model.blocks)

    gs.restore()
    assert [blk.attn for blk in model.blocks] == original_attns

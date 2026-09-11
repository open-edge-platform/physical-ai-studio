# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the graph-safe text/VTC compression index resolver.

``_find_compress_info`` is the core new logic of the graph-safe text model: it
resolves the VTC ``LayerWrapper`` compression span into static Python ints. It
needs only a ``LayerWrapper`` over a dummy layer plus crafted ``input_ids``, so
it runs without instantiating a full Qwen3-VL decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from physicalai.policies.rldx1.components.backbone.graph_safe_text import _find_compress_info
from physicalai.policies.rldx1.components.backbone.layer_wrapper import LayerWrapper

IMG = 151652


class _FakeLM(nn.Module):
    """Minimal container exposing ``.layers`` like a Qwen3-VL text model."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


def _make_lm(internal_projection: int, num_layers: int = 2) -> _FakeLM:
    layers = [
        LayerWrapper(nn.Identity(), layer_idx=i, internal_projection=internal_projection, img_pattern=[IMG])
        for i in range(num_layers)
    ]
    return _FakeLM(layers)


def test_find_compress_info_single_image_span() -> None:
    """begin/end/out_len match the first/last image-token positions."""
    lm = _make_lm(internal_projection=1)
    # image span at indices 2..6 (inclusive), length 5
    ids = torch.tensor([[10, 11, IMG, IMG, IMG, IMG, IMG, 12, 13, 14]])
    n_cog = 4

    info = _find_compress_info(lm, ids, n_cog_tokens=n_cog, num_views=1)

    assert info is not None
    assert info["compress_layer_idx"] == 1
    assert info["static_begin"] == 2
    assert info["static_end"] == 6  # last image-token index (num_views=1 -> m[-1])
    # front[:2] + motion(1) + back[6:] over the LLM ids, plus cog tokens
    assert info["static_out_len"] == 2 + 1 + (ids.shape[1] + n_cog - 6)


def test_find_compress_info_skips_when_begin_ge_end() -> None:
    """A single image token (begin == end) yields no compression."""
    lm = _make_lm(internal_projection=1)
    ids = torch.tensor([[10, 11, IMG, 12, 13]])

    assert _find_compress_info(lm, ids, n_cog_tokens=4, num_views=1) is None


def test_find_compress_info_no_compress_layer() -> None:
    """No layer at its internal_projection -> no compression info."""
    lm = _make_lm(internal_projection=99)  # never matches layer_idx
    ids = torch.tensor([[10, IMG, IMG, IMG, 12]])

    assert _find_compress_info(lm, ids, n_cog_tokens=0, num_views=1) is None

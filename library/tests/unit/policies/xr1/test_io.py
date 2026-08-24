# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 tensor plumbing."""

from __future__ import annotations

import pytest
import torch
from physicalai.policies.xr1.io import (
    continue_text_position_ids,
    build_action_mask,
    build_dit_attention_mask,
    continue_position_ids,
    pad_vector,
    resize_with_pad,
)


class TestPadVector:
    """Padding a dataset's action width up to the configured width."""

    def test_pads_with_zeros(self) -> None:
        """Padding appends zeros on the right."""
        padded = pad_vector(torch.ones(2, 3, 4), 8)

        assert padded.shape == (2, 3, 8)
        assert torch.all(padded[..., 4:] == 0)
        assert torch.all(padded[..., :4] == 1)

    def test_identity_when_exact(self) -> None:
        """An already-wide vector is returned untouched."""
        vector = torch.ones(2, 4)

        assert pad_vector(vector, 4) is vector

    def test_rejects_too_wide(self) -> None:
        """Truncating would silently drop action dimensions."""
        with pytest.raises(ValueError, match="exceeds target width"):
            pad_vector(torch.ones(2, 9), 8)


class TestBuildActionMask:
    """The mask separates real action dimensions from padding."""

    def test_marks_valid_dimensions(self) -> None:
        """Only the dataset's own dimensions are supervised."""
        mask = build_action_mask(torch.zeros(2, 3, 8), valid_dim=5)

        assert torch.all(mask[..., :5] == 1)
        assert torch.all(mask[..., 5:] == 0)

    def test_applies_temporal_mask(self) -> None:
        """Chunks running past the end of an episode are masked out."""
        temporal = torch.tensor([[1, 1, 0], [1, 0, 0]])
        mask = build_action_mask(torch.zeros(2, 3, 8), valid_dim=4, temporal_mask=temporal)

        assert mask.sum() == (2 + 1) * 4

    def test_rejects_valid_dim_over_width(self) -> None:
        """A wider valid_dim than the tensor is a caller bug."""
        with pytest.raises(ValueError, match="exceeds padded action width"):
            build_action_mask(torch.zeros(1, 1, 4), valid_dim=8)


class TestResizeWithPad:
    """Letterboxing keeps the scene geometry undistorted."""

    def test_output_resolution(self) -> None:
        """Images come out at exactly the requested size."""
        assert resize_with_pad(torch.rand(2, 3, 120, 160), 64, 64).shape == (2, 3, 64, 64)

    def test_identity_when_already_sized(self) -> None:
        """No interpolation happens when the size already matches."""
        images = torch.rand(1, 3, 64, 64)

        assert resize_with_pad(images, 64, 64) is images

    def test_pads_rather_than_stretches(self) -> None:
        """A wide image is centred with zero bars, not stretched."""
        images = torch.ones(1, 3, 32, 64)
        resized = resize_with_pad(images, 64, 64)

        assert torch.all(resized[:, :, :16] == 0), "expected a zero bar at the top"
        assert torch.all(resized[:, :, 48:] == 0), "expected a zero bar at the bottom"
        assert torch.all(resized[:, :, 16:48] > 0)

    def test_rejects_wrong_rank(self) -> None:
        """Only batched channel-first tensors are supported."""
        with pytest.raises(ValueError, match="Expected a 4D"):
            resize_with_pad(torch.rand(3, 64, 64), 32, 32)


class TestContinuePositionIds:
    """The action expert continues the backbone's MRoPE grid."""

    def test_starts_after_the_prompt(self) -> None:
        """The first query position is one past the furthest prompt position."""
        vlm_positions = torch.arange(7).view(1, 1, -1).expand(3, 2, 7)
        positions = continue_position_ids(vlm_positions, 5, batch_size=2)

        assert positions.shape == (3, 2, 5)
        assert positions[0, 0].tolist() == [7, 8, 9, 10, 11]

    def test_offsets_predicted_positions(self) -> None:
        """Predicted actions sit further away than the executed prefix."""
        vlm_positions = torch.zeros(3, 1, 4, dtype=torch.long)
        positions = continue_position_ids(
            vlm_positions,
            5,
            batch_size=1,
            suffix_offset=10,
            suffix_length=2,
        )

        assert positions[0, 0].tolist() == [1, 2, 3, 14, 15]


class TestContinueTextPositionIds:
    """The choice head's query tokens continue the language sequence."""

    def test_starts_after_the_prompt(self) -> None:
        """The queries take the next plain text positions."""
        vlm_positions = torch.arange(6).view(1, 1, -1).expand(3, 2, 6)

        positions = continue_text_position_ids(vlm_positions, 4)

        assert positions.shape == (3, 2, 4)
        assert positions[0, 0].tolist() == [6, 7, 8, 9]

    def test_all_axes_advance_together(self) -> None:
        """Text tokens have no separate temporal, row or column position."""
        vlm_positions = torch.zeros(3, 1, 3, dtype=torch.long)
        vlm_positions[1] = 5

        positions = continue_text_position_ids(vlm_positions, 2)

        assert positions[0, 0].tolist() == [1, 2]
        assert positions[1, 0].tolist() == [6, 7]

    def test_mirrors_the_prompt_rank(self) -> None:
        """A text-only prompt can arrive with a single-axis grid."""
        positions = continue_text_position_ids(torch.arange(4).view(1, 1, -1), 2)

        assert positions.shape == (1, 1, 2)


class TestBuildDitAttentionMask:
    """The query attends over the cached prompt and causally over itself."""

    def test_shape_and_causality(self) -> None:
        """Later query tokens see earlier ones, not the reverse."""
        mask = build_dit_attention_mask(torch.ones(2, 7), query_length=4)

        assert mask.shape == (2, 1, 4, 11)
        assert mask.dtype == torch.bool
        query_block = mask[0, 0, :, 7:]
        assert bool(query_block[3, 0]) and not bool(query_block[0, 3])

    def test_respects_prompt_padding(self) -> None:
        """Padded prompt positions stay masked for every query token."""
        cache_mask = torch.tensor([[1, 1, 0, 0]])
        mask = build_dit_attention_mask(cache_mask, query_length=3)

        assert not mask[0, 0, :, 2:4].any()

    def test_drops_prefix_entries_when_masking_enabled(self) -> None:
        """Prefix dropout removes some prefix visibility from later tokens."""
        generator = torch.Generator().manual_seed(0)
        mask = build_dit_attention_mask(
            torch.ones(1, 4),
            query_length=1 + 1 + 8,
            prefix_length=6,
            prefix_mask_prob=1.0,
            state_length=1,
            generator=generator,
        )

        # Action tokens start at index 2 of the query block; with keep_last=2 the
        # first four prefix entries are dropped for tokens after the prefix.
        query_block = mask[0, 0, :, 4:]
        assert not query_block[8, 2:6].any()

    def test_no_dropout_when_probability_zero(self) -> None:
        """With masking disabled the prefix stays fully visible."""
        mask = build_dit_attention_mask(
            torch.ones(1, 4),
            query_length=1 + 1 + 8,
            prefix_length=6,
            prefix_mask_prob=0.0,
        )

        assert mask[0, 0, 8, 4:10].all()

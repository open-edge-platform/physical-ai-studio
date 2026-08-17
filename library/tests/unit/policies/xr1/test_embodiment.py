# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR1's fixed slot layout and the dataset-to-slot mapping."""

from __future__ import annotations

import pytest
import torch
from physicalai.policies.xr1.embodiment import (
    ACTION_SLOTS,
    ALOHA_STATE_TO_XR1,
    STATE_SLOTS,
    XR1_ACTION_DIM,
    XR1_STATE_DIM,
    gather_slots,
    scatter_slots,
    slot_mask,
    validate_slot_map,
)

ALOHA_DIM = 14


class TestReferenceLayout:
    """The slot layout is a contract with the released checkpoints."""

    def test_slots_do_not_overlap(self) -> None:
        """Two parts sharing a slot would silently overwrite each other."""
        for name, slots in (("state", STATE_SLOTS), ("action", ACTION_SLOTS)):
            occupied: set[int] = set()
            for part, (start, end) in slots.items():
                indices = set(range(start, end))
                assert not occupied & indices, f"{name} part {part} overlaps an earlier part"
                occupied |= indices

    def test_slots_fit_the_vector(self) -> None:
        """Every part must land inside the 60 slots the checkpoints allocate."""
        assert max(end for _, end in STATE_SLOTS.values()) <= XR1_STATE_DIM
        assert max(end for _, end in ACTION_SLOTS.values()) <= XR1_ACTION_DIM


class TestAlohaStateMap:
    """The one mapping that can be stated exactly."""

    def test_covers_every_aloha_dimension(self) -> None:
        """ALOHA state is 6 joints plus a gripper per arm."""
        assert len(ALOHA_STATE_TO_XR1) == ALOHA_DIM

    def test_grippers_land_in_the_gripper_slots(self) -> None:
        """Slot 7 and slot 15 are grippers; the seventh joint slots stay empty."""
        assert ALOHA_STATE_TO_XR1[6] == STATE_SLOTS["left_gripper"][0]
        assert ALOHA_STATE_TO_XR1[13] == STATE_SLOTS["right_gripper"][0]

    def test_arms_land_in_the_arm_slots(self) -> None:
        """Each arm's six joints occupy the first six of its seven joint slots."""
        left_start, left_end = STATE_SLOTS["left_arm"]
        right_start, right_end = STATE_SLOTS["right_arm"]

        assert ALOHA_STATE_TO_XR1[:6] == tuple(range(left_start, left_start + 6))
        assert ALOHA_STATE_TO_XR1[7:13] == tuple(range(right_start, right_start + 6))
        assert left_end - 1 not in ALOHA_STATE_TO_XR1
        assert right_end - 1 not in ALOHA_STATE_TO_XR1

    def test_scatter_matches_the_map(self) -> None:
        """Routing a 14-vector must put each entry in exactly its named slot."""
        vector = torch.arange(1, ALOHA_DIM + 1, dtype=torch.float32)[None]

        out = scatter_slots(vector, ALOHA_STATE_TO_XR1, XR1_STATE_DIM)

        assert out.shape == (1, XR1_STATE_DIM)
        assert out[0, 7] == 7.0  # left gripper
        assert out[0, 6] == 0.0  # unused seventh joint
        assert out[0, 15] == 14.0  # right gripper
        assert out[0, 16:].sum() == 0.0


class TestSlotRouting:
    """Generic scatter and gather behaviour."""

    def test_gather_inverts_scatter(self) -> None:
        """The round trip must be lossless for the mapped dimensions."""
        vector = torch.randn(2, 3, ALOHA_DIM)

        routed = scatter_slots(vector, ALOHA_STATE_TO_XR1, XR1_STATE_DIM)

        assert torch.equal(gather_slots(routed, ALOHA_STATE_TO_XR1), vector)

    def test_preserves_leading_dimensions(self) -> None:
        """Chunks are ``(batch, horizon, dim)``, so only the last axis changes."""
        vector = torch.randn(4, 7, 2)

        out = scatter_slots(vector, (5, 9), 32)

        assert out.shape == (4, 7, 32)

    def test_rejects_a_width_mismatch(self) -> None:
        """A silently truncated state would poison every downstream tensor."""
        with pytest.raises(ValueError, match="does not match a slot map"):
            scatter_slots(torch.randn(1, 5), ALOHA_STATE_TO_XR1, XR1_STATE_DIM)

    def test_mask_marks_only_the_mapped_slots(self) -> None:
        """Unmapped slots carry no supervision."""
        mask = slot_mask((0, 3, 9), 12)

        assert mask.sum() == 3
        assert mask[0] and mask[3] and mask[9]


class TestValidation:
    """A bad map must fail at config time, not inside a forward pass."""

    def test_rejects_empty(self) -> None:
        """An empty map maps nothing and is always a mistake."""
        with pytest.raises(ValueError, match="must not be empty"):
            validate_slot_map((), 32, "state_slot_map")

    def test_rejects_duplicates(self) -> None:
        """Two dimensions in one slot means one of them is lost."""
        with pytest.raises(ValueError, match="two dimensions to the same slot"):
            validate_slot_map((0, 1, 1), 32, "state_slot_map")

    def test_rejects_out_of_range(self) -> None:
        """A slot past the vector width would raise deep inside index_copy."""
        with pytest.raises(ValueError, match="fall outside the 32 available slots"):
            validate_slot_map((0, 32), 32, "state_slot_map")

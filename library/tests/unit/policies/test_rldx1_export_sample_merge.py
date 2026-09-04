# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for export-sample key trimming helper behavior."""

from __future__ import annotations

from physicalai.policies.rldx1.export_helpers import trim_export_sample

import torch


def test_trim_export_sample_keeps_live_observation_tensors() -> None:
    """trim_export_sample preserves live values for known export input keys."""

    live_input = {
        "pixel_values": torch.full((1, 1, 2, 2), 3.0),
        "state": torch.full((1, 4), 4.0),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "position_ids": torch.zeros(3, 1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    trimmed = trim_export_sample(live_input)

    assert trimmed is not None
    assert torch.equal(trimmed["pixel_values"], live_input["pixel_values"])
    assert torch.equal(trimmed["state"], live_input["state"])
    assert torch.equal(trimmed["input_ids"], live_input["input_ids"])
    assert torch.equal(trimmed["position_ids"], live_input["position_ids"])
    assert torch.equal(trimmed["attention_mask"], live_input["attention_mask"])


def test_trim_export_sample_without_export_sample_uses_input() -> None:
    """Trimming is a pure helper key filter over known export input names."""

    live_input = {
        "pixel_values": torch.full((1, 1, 2, 2), 1.0),
        "state": torch.full((1, 4), 2.0),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }

    trimmed = trim_export_sample(live_input)

    assert trimmed is not None
    assert set(trimmed) == {"pixel_values", "state", "input_ids"}
    assert torch.equal(trimmed["pixel_values"], live_input["pixel_values"])
    assert torch.equal(trimmed["state"], live_input["state"])

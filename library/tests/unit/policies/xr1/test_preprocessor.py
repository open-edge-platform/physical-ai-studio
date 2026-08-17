# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR1 pre- and postprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from physicalai.policies.xr1 import XR1Config, make_xr1_preprocessors
from physicalai.policies.xr1.preprocessor import XR1Preprocessor, normalization_map, split_features

if TYPE_CHECKING:
    from physicalai.data import Feature

STATE_DIM = 8
ACTION_DIM = 8


class TestNormalizationMap:
    """Normalization mode plumbing."""

    def test_known_modes(self) -> None:
        """Both documented modes resolve to a strategy per feature type."""
        assert len(normalization_map("MEAN_STD")) == 2
        assert len(normalization_map("QUANTILES")) == 2

    def test_unknown_mode(self) -> None:
        """An unknown mode fails at construction, not at the first batch."""
        with pytest.raises(ValueError, match="Unsupported normalization_mode"):
            normalization_map("ROBUST")


class TestSplitFeatures:
    """Feature partitioning."""

    def test_splits_state_and_action(self, dataset_features: dict[str, Feature]) -> None:
        """State and action features are separated by type."""
        state, action = split_features(dataset_features)

        assert set(state) == {"state"}
        assert set(action) == {"action"}

    def test_handles_missing_schema(self) -> None:
        """No schema means no normalization, which is valid in tests."""
        assert split_features(None) == ({}, {})


class TestPreprocessor:
    """Observation to backbone inputs."""

    def test_produces_model_inputs(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        observation_batch: dict[str, Any],
    ) -> None:
        """Every tensor the model needs is present and correctly shaped."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        processed = preprocessor(observation_batch)

        assert processed["state"].shape == (2, tiny_config.state_len, tiny_config.max_state_dim)
        assert processed["action"].shape == (2, tiny_config.chunk_size, tiny_config.max_action_dim)
        assert processed["action_mask"].shape == processed["action"].shape
        assert processed["input_ids"].shape[0] == 2
        assert "pixel_values" in processed

    def test_pads_narrow_state_and_action(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        observation_batch: dict[str, Any],
    ) -> None:
        """A 5-dim state and 6-dim action pad up, and the mask records the real width."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        processed = preprocessor(observation_batch)

        assert torch.all(processed["state"][..., 5:] == 0)
        assert processed["action_mask"][0, 0].tolist() == [1.0] * 6 + [0.0] * 2

    def test_orders_images_by_configured_views(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
    ) -> None:
        """Views are consumed in config order, not dict order."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)
        batch = {
            "state": torch.randn(1, 5),
            "images": {"wrist": torch.ones(1, 3, 32, 32), "top": torch.zeros(1, 3, 32, 32)},
        }

        views = preprocessor._ordered_images(batch)  # noqa: SLF001 - ordering is the behavior under test

        assert float(views[0].max()) == 0.0, "expected the 'top' view first"
        assert float(views[1].max()) > 0.0

    def test_falls_back_to_dataset_view_names(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
    ) -> None:
        """A dataset whose cameras are named differently still trains."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)
        batch = {"state": torch.randn(1, 5), "images": {"cam_high": torch.rand(1, 3, 32, 32)}}

        assert len(preprocessor._ordered_images(batch)) == 1  # noqa: SLF001 - fallback is the behavior

    def test_scales_uint8_images(self, tiny_config: XR1Config, stub_processor: Any) -> None:
        """Byte images are scaled to [0, 1] rather than fed raw."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)
        batch = {"state": torch.randn(1, 5), "images": {"top": torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)}}

        views = preprocessor._ordered_images(batch)  # noqa: SLF001 - dtype handling is the behavior

        assert float(views[0].max()) == pytest.approx(1.0)

    def test_prompt_names_each_view(self, tiny_config: XR1Config, stub_processor: Any) -> None:
        """The prompt announces views so the backbone can tell cameras apart."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        prompt = preprocessor.build_prompt("transfer the cube", 2)

        assert "top view:" in prompt
        assert "wrist view:" in prompt
        assert "transfer the cube" in prompt

    def test_requires_state(self, tiny_config: XR1Config, stub_processor: Any) -> None:
        """XR1 is state-conditioned; a missing state is a data error."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        with pytest.raises(KeyError, match="requires a state"):
            preprocessor({"images": {"top": torch.rand(1, 3, 32, 32)}})

    def test_action_is_optional(self, tiny_config: XR1Config, stub_processor: Any) -> None:
        """Inference batches carry no action."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        processed = preprocessor({"state": torch.randn(1, 5), "task": "do it"})

        assert "action" not in processed

    def test_normalizes_when_given_statistics(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        dataset_features: dict[str, Feature],
        observation_batch: dict[str, Any],
    ) -> None:
        """With unit statistics normalization is a no-op, which pins the wiring."""
        preprocessor = XR1Preprocessor(tiny_config, dataset_features, stub_processor)

        processed = preprocessor(observation_batch)

        assert torch.allclose(processed["state"][:, 0, :5], observation_batch["state"], atol=1e-5)


class TestPostprocessor:
    """Predicted actions back to dataset units."""

    def test_trims_padding(
        self,
        tiny_config: XR1Config,
        dataset_features: dict[str, Feature],
    ) -> None:
        """Padded dimensions are dropped so the robot sees its own action width."""
        _, postprocessor = make_xr1_preprocessors(tiny_config, dataset_features)

        result = postprocessor({"action": torch.randn(2, tiny_config.chunk_size, ACTION_DIM)})

        assert result["action"].shape == (2, tiny_config.chunk_size, 6)

    def test_passthrough_without_action(self, tiny_config: XR1Config) -> None:
        """A batch with no action is returned unchanged."""
        _, postprocessor = make_xr1_preprocessors(tiny_config, None)

        assert postprocessor({"other": 1}) == {"other": 1}

    def test_roundtrip_is_identity_with_unit_stats(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        dataset_features: dict[str, Feature],
        observation_batch: dict[str, Any],
    ) -> None:
        """Normalizing then denormalizing recovers the original actions."""
        preprocessor, postprocessor = make_xr1_preprocessors(tiny_config, dataset_features, stub_processor)

        processed = preprocessor(observation_batch)
        recovered = postprocessor({"action": processed["action"]})["action"]

        assert torch.allclose(recovered, observation_batch["action"], atol=1e-5)

class TestDeviceConsistency:
    """Everything the model consumes must land on the batch's device.

    The Qwen3-VL processor always returns CPU tensors. Passing them straight through
    fails at the first embedding lookup with "Expected all tensors to be on the same
    device", which is only visible once training runs on an accelerator.
    """

    def test_outputs_follow_the_state_device(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        observation_batch: dict[str, Any],
    ) -> None:
        """Every returned tensor shares the state tensor's device."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)

        processed = preprocessor(observation_batch)

        expected = processed["state"].device
        for name, value in processed.items():
            if isinstance(value, torch.Tensor):
                assert value.device == expected, f"{name} is on {value.device}, expected {expected}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_outputs_move_to_cuda(
        self,
        tiny_config: XR1Config,
        stub_processor: Any,
        observation_batch: dict[str, Any],
    ) -> None:
        """Tokens produced on the CPU are moved onto the accelerator."""
        preprocessor = XR1Preprocessor(tiny_config, None, stub_processor)
        batch = {
            **observation_batch,
            "state": observation_batch["state"].cuda(),
            "action": observation_batch["action"].cuda(),
        }

        processed = preprocessor(batch)

        assert processed["input_ids"].is_cuda
        assert processed["attention_mask"].is_cuda
        assert processed["pixel_values"].is_cuda

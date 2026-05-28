# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Pi05 Real-Time Chunking (RTC) functionality.

Tests the RTC guidance correction that improves temporal consistency
of action predictions by blending current predictions with the previous
action chunk during the denoising loop.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

if TYPE_CHECKING:
    from physicalai.policies.pi05.model import Pi05Model


# ============================================================================ #
# _compute_prefix_weights Tests                                                #
# ============================================================================ #


class TestComputePrefixWeights:
    """Tests for Pi05Model._compute_prefix_weights."""

    @staticmethod
    def _call_compute_prefix_weights(
        inference_delay: Tensor,
        execution_horizon: Tensor,
        chunk_size: int = 50,
        prefix_attention_schedule: str = "linear",
    ) -> Tensor:
        """Call _compute_prefix_weights via the unbound method on a stub."""
        from physicalai.policies.pi05.model import Pi05Model

        class _Stub:
            _chunk_size = chunk_size

        return Pi05Model._compute_prefix_weights(
            _Stub(),  # type: ignore[arg-type]
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prefix_attention_schedule=prefix_attention_schedule,
        )

    def test_output_shape(self) -> None:
        """Weights have shape (1, chunk_size, 1)."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(4),
            execution_horizon=torch.tensor(10),
            chunk_size=50,
        )
        assert weights.shape == (1, 50, 1)

    def test_weights_bounded_zero_one(self) -> None:
        """All weights are in [0, 1]."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(5),
            execution_horizon=torch.tensor(20),
            chunk_size=50,
        )
        assert (weights >= 0.0).all()
        assert (weights <= 1.0).all()

    def test_weights_monotonically_decreasing(self) -> None:
        """Weights decrease along the chunk dimension."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(2),
            execution_horizon=torch.tensor(15),
            chunk_size=50,
        )
        w = weights.squeeze()
        # Non-zero region should be monotonically non-increasing
        nonzero_mask = w > 0
        nonzero_weights = w[nonzero_mask]
        if len(nonzero_weights) > 1:
            diffs = nonzero_weights[1:] - nonzero_weights[:-1]
            assert (diffs <= 1e-6).all()

    def test_weights_zero_beyond_execution_horizon(self) -> None:
        """Weights are zero for indices >= execution_horizon."""
        execution_horizon = 10
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(0),
            execution_horizon=torch.tensor(execution_horizon),
            chunk_size=50,
        )
        w = weights.squeeze()
        # At index == execution_horizon, weight = (end - idx) / denom = 0 / denom = 0
        assert (w[execution_horizon:] == 0.0).all()

    def test_inference_delay_clamped_to_execution_horizon(self) -> None:
        """If inference_delay > execution_horizon, start is clamped."""
        weights_large_delay = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(100),
            execution_horizon=torch.tensor(10),
            chunk_size=50,
        )
        weights_at_horizon = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(10),
            execution_horizon=torch.tensor(10),
            chunk_size=50,
        )
        torch.testing.assert_close(weights_large_delay, weights_at_horizon)

    def test_linear_schedule(self) -> None:
        """Linear schedule produces linearly decaying weights."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(0),
            execution_horizon=torch.tensor(10),
            chunk_size=10,
            prefix_attention_schedule="linear",
        )
        w = weights.squeeze()
        # Formula: (end - idx) / (end - start + 1) = (10 - idx) / 11
        expected = torch.clamp(
            (10.0 - torch.arange(10, dtype=torch.float32)) / 11.0,
            min=0.0,
            max=1.0,
        )
        torch.testing.assert_close(w, expected)

    def test_exp_schedule_differs_from_linear(self) -> None:
        """Exponential schedule differs from linear schedule."""
        weights_linear = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(2),
            execution_horizon=torch.tensor(10),
            chunk_size=50,
            prefix_attention_schedule="linear",
        )
        weights_exp = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(2),
            execution_horizon=torch.tensor(10),
            chunk_size=50,
            prefix_attention_schedule="exp",
        )
        assert not torch.allclose(weights_linear, weights_exp)

    def test_exp_schedule_bounded(self) -> None:
        """Exponential schedule weights are still in [0, 1]."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(3),
            execution_horizon=torch.tensor(15),
            chunk_size=50,
            prefix_attention_schedule="exp",
        )
        assert (weights >= 0.0).all()
        assert (weights <= 1.0).all()

    def test_exp_schedule_formula(self) -> None:
        """Verify exponential schedule transformation: w * (exp(w) - 1) / (e - 1)."""
        weights_linear = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(0),
            execution_horizon=torch.tensor(5),
            chunk_size=5,
            prefix_attention_schedule="linear",
        )
        weights_exp = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(0),
            execution_horizon=torch.tensor(5),
            chunk_size=5,
            prefix_attention_schedule="exp",
        )
        # Manually compute expected exp weights
        w_lin = weights_linear.squeeze()
        expected_exp = w_lin * (torch.exp(w_lin) - 1.0) / (math.e - 1.0)
        torch.testing.assert_close(weights_exp.squeeze(), expected_exp)

    def test_zero_execution_horizon(self) -> None:
        """Zero execution horizon produces all-zero weights."""
        weights = self._call_compute_prefix_weights(
            inference_delay=torch.tensor(0),
            execution_horizon=torch.tensor(0),
            chunk_size=50,
        )
        w = weights.squeeze()
        assert (w == 0.0).all()


# ============================================================================ #
# _rtc_correct Tests                                                           #
# ============================================================================ #


class TestRtcCorrect:
    """Tests for Pi05Model._rtc_correct."""

    @staticmethod
    def _call_rtc_correct(
        x_t: Tensor,
        v_t: Tensor,
        prev_chunk_left_over: Tensor,
        prefix_weights: Tensor,
        time: float,
        max_guidance_weight: Tensor,
    ) -> Tensor:
        """Call _rtc_correct via the class-level function."""
        from physicalai.policies.pi05.model import Pi05Model

        # Note: Pi05Model._rtc_correct's signature does not include 'self'
        # due to base class wrapping, so call it directly.
        return Pi05Model._rtc_correct(
            x_t, v_t, prev_chunk_left_over, prefix_weights, time, max_guidance_weight,
        )

    def test_output_shape_matches_v_t(self) -> None:
        """Output shape matches v_t shape."""
        bsize, chunk_size, action_dim = 1, 50, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.ones(1, chunk_size, 1) * 0.5
        max_gw = torch.tensor(10.0)

        result = self._call_rtc_correct(x_t, v_t, prev_chunk, prefix_weights, time=0.5, max_guidance_weight=max_gw)
        assert result.shape == v_t.shape

    def test_zero_prefix_weights_no_correction(self) -> None:
        """Zero prefix weights means no correction applied."""
        bsize, chunk_size, action_dim = 1, 50, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.zeros(1, chunk_size, 1)
        max_gw = torch.tensor(10.0)

        result = self._call_rtc_correct(x_t, v_t, prev_chunk, prefix_weights, time=0.5, max_guidance_weight=max_gw)
        torch.testing.assert_close(result, v_t)

    def test_correction_modifies_velocity(self) -> None:
        """Non-zero prefix weights result in modified velocity."""
        bsize, chunk_size, action_dim = 1, 50, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.ones(1, chunk_size, 1) * 0.8
        max_gw = torch.tensor(10.0)

        result = self._call_rtc_correct(x_t, v_t, prev_chunk, prefix_weights, time=0.5, max_guidance_weight=max_gw)
        assert not torch.allclose(result, v_t)

    def test_guidance_weight_capped_by_max(self) -> None:
        """Guidance weight is capped by max_guidance_weight."""
        bsize, chunk_size, action_dim = 1, 10, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.ones(1, chunk_size, 1)

        # With a very small max guidance, the correction should be small
        result_small = self._call_rtc_correct(
            x_t, v_t, prev_chunk, prefix_weights, time=0.5, max_guidance_weight=torch.tensor(0.1),
        )
        # With a large max guidance, the correction should be larger
        result_large = self._call_rtc_correct(
            x_t, v_t, prev_chunk, prefix_weights, time=0.5, max_guidance_weight=torch.tensor(100.0),
        )
        # The difference from original v_t should be larger with larger max_gw
        diff_small = (result_small - v_t).abs().sum()
        diff_large = (result_large - v_t).abs().sum()
        assert diff_large > diff_small

    def test_time_near_one_large_correction(self) -> None:
        """At time near 1 (early denoising), guidance is stronger."""
        bsize, chunk_size, action_dim = 1, 10, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.ones(1, chunk_size, 1)
        max_gw = torch.tensor(50.0)

        # At time=0.9 (early), guidance should be strong
        result_early = self._call_rtc_correct(
            x_t, v_t, prev_chunk, prefix_weights, time=0.9, max_guidance_weight=max_gw,
        )
        # At time=0.1 (late), guidance should be weaker
        result_late = self._call_rtc_correct(
            x_t, v_t, prev_chunk, prefix_weights, time=0.1, max_guidance_weight=max_gw,
        )
        diff_early = (result_early - v_t).abs().sum()
        diff_late = (result_late - v_t).abs().sum()
        assert diff_early > diff_late

    def test_prev_chunk_matches_prediction_no_correction(self) -> None:
        """If prev_chunk == predicted clean actions, error is zero → no correction."""
        bsize, chunk_size, action_dim = 1, 10, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        time = 0.5
        # Predicted clean actions: x1_t = x_t - time * v_t
        x1_t = x_t - time * v_t
        # If prev_chunk equals the prediction, err = 0
        prev_chunk = x1_t.clone()
        prefix_weights = torch.ones(1, chunk_size, 1)
        max_gw = torch.tensor(10.0)

        result = self._call_rtc_correct(x_t, v_t, prev_chunk, prefix_weights, time=time, max_guidance_weight=max_gw)
        torch.testing.assert_close(result, v_t, atol=1e-5, rtol=1e-5)

    def test_no_nan_or_inf_in_output(self) -> None:
        """Output should not contain NaN or Inf for typical inputs."""
        bsize, chunk_size, action_dim = 1, 10, 7
        x_t = torch.randn(bsize, chunk_size, action_dim)
        v_t = torch.randn(bsize, chunk_size, action_dim)
        prev_chunk = torch.randn(bsize, chunk_size, action_dim)
        prefix_weights = torch.ones(1, chunk_size, 1)
        max_gw = torch.tensor(10.0)

        # Test across various time values including edge cases
        for time in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = self._call_rtc_correct(x_t, v_t, prev_chunk, prefix_weights, time=time, max_guidance_weight=max_gw)
            assert not torch.isnan(result).any(), f"NaN at time={time}"
            assert not torch.isinf(result).any(), f"Inf at time={time}"


# ============================================================================ #
# sample_input with enable_rtc Tests                                           #
# ============================================================================ #


class TestSampleInputRtc:
    """Tests for Pi05.sample_input property with enable_rtc=True."""

    @staticmethod
    def _call_sample_input_rtc(dataset_stats: dict, chunk_size: int = 50, max_action_dim: int = 32) -> dict:
        """Invoke the Pi05.sample_input property with enable_rtc=True on a minimal stub."""
        from physicalai.policies.pi05 import Pi05

        class _ModelStub:
            enable_rtc = True
            paligemma_with_expert = torch.nn.Linear(1, 1)

        class _ConfigStub:
            def __init__(self) -> None:
                self.chunk_size = chunk_size
                self.max_action_dim = max_action_dim

        class _Stub:
            def __init__(self) -> None:
                self._dataset_stats = dataset_stats
                self.model = _ModelStub()
                self.config = _ConfigStub()

        stub = _Stub()
        stub.inputs_schema = Pi05.inputs_schema.fget(stub)  # type: ignore[attr-defined]
        return Pi05.sample_input.fget(stub)  # type: ignore[attr-defined]

    def test_contains_rtc_keys(self) -> None:
        """RTC sample input contains the four RTC-specific keys."""
        stats = {
            "observation.state": {"name": "state", "shape": (8,), "type": "STATE"},
            "observation.image": {"name": "image", "shape": (3, 224, 224), "type": "VISUAL"},
        }
        sample_input = self._call_sample_input_rtc(stats)
        assert "prev_chunk_left_over" in sample_input
        assert "inference_delay" in sample_input
        assert "max_guidance_weight" in sample_input
        assert "execution_horizon" in sample_input

    def test_prev_chunk_left_over_shape(self) -> None:
        """prev_chunk_left_over has shape (1, chunk_size, max_action_dim)."""
        stats = {
            "observation.state": {"name": "state", "shape": (8,), "type": "STATE"},
            "observation.image": {"name": "image", "shape": (3, 224, 224), "type": "VISUAL"},
        }
        chunk_size, max_action_dim = 50, 32
        sample_input = self._call_sample_input_rtc(stats, chunk_size=chunk_size, max_action_dim=max_action_dim)
        assert sample_input["prev_chunk_left_over"].shape == (1, chunk_size, max_action_dim)

    def test_rtc_scalar_dtypes(self) -> None:
        """RTC scalar inputs have correct dtypes."""
        stats = {
            "observation.state": {"name": "state", "shape": (8,), "type": "STATE"},
            "observation.image": {"name": "image", "shape": (3, 224, 224), "type": "VISUAL"},
        }
        sample_input = self._call_sample_input_rtc(stats)
        assert sample_input["inference_delay"].dtype == torch.long
        assert sample_input["max_guidance_weight"].dtype == torch.float32
        assert sample_input["execution_horizon"].dtype == torch.long
        assert sample_input["prev_chunk_left_over"].dtype == torch.float32

    def test_also_contains_standard_keys(self) -> None:
        """RTC sample input also contains the standard model inputs."""
        from physicalai.data.observation import IMAGES, STATE, TASK

        stats = {
            "observation.state": {"name": "state", "shape": (8,), "type": "STATE"},
            "observation.image": {"name": "image", "shape": (3, 224, 224), "type": "VISUAL"},
        }
        sample_input = self._call_sample_input_rtc(stats)
        assert STATE in sample_input
        assert IMAGES in sample_input
        assert TASK in sample_input


# ============================================================================ #
# sample_actions RTC Integration Tests                                         #
# ============================================================================ #


class TestSampleActionsRTC:
    """Integration tests for sample_actions with and without RTC.

    Uses smallest model variants (gemma_300m) with patched projection_dim
    to keep memory usage low in CI (~300M params instead of ~2.6B).
    """

    @pytest.fixture(scope="class")
    def model(self) -> "Pi05Model":
        """Create a small Pi05Model for testing."""
        from physicalai.policies.pi05.model import PaliGemmaWithExpertModel, Pi05Model

        original_init = PaliGemmaWithExpertModel.__init__

        def _patched_init(self_inner, *args, **kwargs):
            original_init(self_inner, *args, **kwargs)
            # After construction, resize multi_modal_projector to match gemma_300m width (1024)
            # The projector output must match text_config.hidden_size for embed_prefix concat
            projector = self_inner.paligemma.model.multi_modal_projector
            in_features = projector.linear.in_features if hasattr(projector, "linear") else projector.in_features
            hidden_size = self_inner.paligemma.config.text_config.hidden_size
            import torch.nn as nn

            if hasattr(projector, "linear"):
                projector.linear = nn.Linear(in_features, hidden_size)
            else:
                self_inner.paligemma.model.multi_modal_projector = nn.Linear(in_features, hidden_size)

        dataset_stats = {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
            "observation.image": {
                "name": "observation.image",
                "shape": (3, 224, 224),
                "type": "VISUAL",
            },
        }
        with patch.object(PaliGemmaWithExpertModel, "__init__", _patched_init):
            model = Pi05Model(
                dataset_stats=dataset_stats,
                paligemma_variant="gemma_300m",
                action_expert_variant="gemma_300m",
                dtype="float32",
                chunk_size=50,
                max_action_dim=32,
                num_inference_steps=2,  # Use few steps for faster tests
            )
        model.eval()
        return model

    @pytest.fixture()
    def sample_batch(self, model) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Create a minimal batch for sample_actions."""
        device = next(model.parameters()).device
        # embed_prefix expects (num_cameras, batch, C, H, W) stacked tensor
        images = torch.randn(1, 1, 3, 224, 224, device=device)
        img_masks = torch.ones(1, 1, dtype=torch.bool, device=device)
        # Minimal token/mask setup
        tokens = torch.ones(1, 10, dtype=torch.long, device=device)
        masks = torch.ones(1, 10, dtype=torch.long, device=device)
        return images, img_masks, tokens, masks

    def test_rtc_disabled_when_no_prev_chunk(self, model, sample_batch) -> None:
        """Without prev_chunk, sample_actions behaves identically regardless of RTC params."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)

        with torch.no_grad():
            actions_no_rtc = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
            )
            actions_with_rtc_params = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=10.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=None,
            )

        torch.testing.assert_close(actions_no_rtc, actions_with_rtc_params)

    def test_rtc_modifies_output_with_prev_chunk(self, model, sample_batch) -> None:
        """With prev_chunk provided, RTC modifies the denoised actions."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)
        prev_chunk = torch.randn(1, 50, 32, device=device)

        with torch.no_grad():
            actions_no_rtc = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
            )
            actions_with_rtc = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=10.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )

        assert not torch.allclose(actions_no_rtc, actions_with_rtc, atol=1e-4)

    def test_rtc_output_shape_unchanged(self, model, sample_batch) -> None:
        """RTC does not change the output shape."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)
        prev_chunk = torch.randn(1, 50, 32, device=device)

        with torch.no_grad():
            actions = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=10.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )

        assert actions.shape == (1, 50, 32)

    def test_rtc_zero_guidance_weight_minimal_effect(self, model, sample_batch) -> None:
        """With max_guidance_weight=0, RTC correction should be zero."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)
        prev_chunk = torch.randn(1, 50, 32, device=device)

        with torch.no_grad():
            actions_no_rtc = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
            )
            actions_zero_gw = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=0.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )

        # With 0 max guidance weight, correction = v_t - 0 * correction = v_t
        torch.testing.assert_close(actions_no_rtc, actions_zero_gw)

    def test_rtc_stronger_guidance_larger_difference(self, model, sample_batch) -> None:
        """Larger max_guidance_weight produces larger deviation from unguided output."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)
        prev_chunk = torch.randn(1, 50, 32, device=device)

        with torch.no_grad():
            actions_no_rtc = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
            )
            actions_weak = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=1.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )
            actions_strong = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=50.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )

        diff_weak = (actions_weak - actions_no_rtc).abs().sum()
        diff_strong = (actions_strong - actions_no_rtc).abs().sum()
        assert diff_strong > diff_weak

    def test_rtc_no_nan_inf(self, model, sample_batch) -> None:
        """RTC output contains no NaN or Inf values."""
        images, img_masks, tokens, masks = sample_batch
        device = next(model.parameters()).device
        noise = model.sample_noise((1, 50, 32), device)
        prev_chunk = torch.randn(1, 50, 32, device=device)

        with torch.no_grad():
            actions = model.sample_actions(
                images, img_masks, tokens, masks,
                noise=noise.clone(),
                rtc_max_guidance=10.0,
                rtc_execution_horizon=10,
                rtc_latency=5.0,
                rtc_prev_action_chunk=prev_chunk,
            )

        assert not torch.isnan(actions).any()
        assert not torch.isinf(actions).any()


# ============================================================================ #
# predict_action_chunk RTC Integration Tests                                   #
# ============================================================================ #


class TestPredictActionChunkRTC:
    """Tests for predict_action_chunk with RTC batch keys."""

    @pytest.fixture(scope="class")
    def model(self) -> "Pi05Model":
        """Create a small Pi05Model for testing."""
        from physicalai.policies.pi05.model import PaliGemmaWithExpertModel, Pi05Model

        original_init = PaliGemmaWithExpertModel.__init__

        def _patched_init(self_inner, *args, **kwargs):
            original_init(self_inner, *args, **kwargs)
            projector = self_inner.paligemma.model.multi_modal_projector
            in_features = projector.linear.in_features if hasattr(projector, "linear") else projector.in_features
            hidden_size = self_inner.paligemma.config.text_config.hidden_size
            import torch.nn as nn

            if hasattr(projector, "linear"):
                projector.linear = nn.Linear(in_features, hidden_size)
            else:
                self_inner.paligemma.model.multi_modal_projector = nn.Linear(in_features, hidden_size)

        dataset_stats = {
            "observation.state": {
                "name": "observation.state",
                "shape": (8,),
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
                "q01": [-1.0] * 8,
                "q99": [1.0] * 8,
            },
            "action": {
                "name": "action",
                "shape": (7,),
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
            },
            "observation.image": {
                "name": "observation.image",
                "shape": (3, 224, 224),
                "type": "VISUAL",
            },
        }
        with patch.object(PaliGemmaWithExpertModel, "__init__", _patched_init):
            model = Pi05Model(
                dataset_stats=dataset_stats,
                paligemma_variant="gemma_300m",
                action_expert_variant="gemma_300m",
                dtype="float32",
                chunk_size=50,
                max_action_dim=32,
                num_inference_steps=2,
            )
        model.eval()
        return model

    @staticmethod
    def _make_batch(model, device) -> dict:
        """Create a preprocessed-style batch dict."""
        from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
        from physicalai.data.observation import IMAGES

        return {
            # Preprocessor stacks images into (num_cameras, batch, C, H, W)
            IMAGES: torch.randn(1, 1, 3, 224, 224, device=device),
            IMAGE_MASKS: torch.ones(1, 1, dtype=torch.bool, device=device),
            TOKENIZED_PROMPT: torch.ones(1, 10, dtype=torch.long, device=device),
            TOKENIZED_PROMPT_MASK: torch.ones(1, 10, dtype=torch.long, device=device),
        }

    def test_predict_action_chunk_without_rtc_keys(self, model) -> None:
        """predict_action_chunk works without RTC keys in batch."""
        device = next(model.parameters()).device
        batch = self._make_batch(model, device)

        with torch.no_grad():
            actions = model.predict_action_chunk(batch)

        assert actions.shape[0] == 1
        assert actions.shape[2] == 7  # Original action dim, unpadded

    def test_predict_action_chunk_with_rtc_keys(self, model) -> None:
        """predict_action_chunk uses RTC keys from batch when present."""
        device = next(model.parameters()).device
        batch = self._make_batch(model, device)

        # Add RTC keys to batch
        batch["prev_chunk_left_over"] = torch.randn(1, 50, 32, device=device)
        batch["inference_delay"] = 4.0
        batch["max_guidance_weight"] = 10.0
        batch["execution_horizon"] = 10

        with torch.no_grad():
            actions = model.predict_action_chunk(batch)

        assert actions.shape[0] == 1
        assert actions.shape[2] == 7
        assert not torch.isnan(actions).any()

    def test_predict_action_chunk_rtc_vs_no_rtc(self, model) -> None:
        """predict_action_chunk produces different results with vs without RTC batch keys."""
        from physicalai.data.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
        from physicalai.data.observation import IMAGES

        device = next(model.parameters()).device
        batch_no_rtc = self._make_batch(model, device)
        batch_with_rtc = self._make_batch(model, device)

        # Ensure same images/tokens
        batch_with_rtc[IMAGES] = batch_no_rtc[IMAGES].clone()
        batch_with_rtc[TOKENIZED_PROMPT] = batch_no_rtc[TOKENIZED_PROMPT].clone()
        batch_with_rtc[TOKENIZED_PROMPT_MASK] = batch_no_rtc[TOKENIZED_PROMPT_MASK].clone()

        # Add RTC keys
        batch_with_rtc["prev_chunk_left_over"] = torch.randn(1, 50, 32, device=device)
        batch_with_rtc["inference_delay"] = 4.0
        batch_with_rtc["max_guidance_weight"] = 10.0
        batch_with_rtc["execution_horizon"] = 10

        # Need to fix the noise for fair comparison — monkey-patch sample_noise
        fixed_noise = model.sample_noise((1, 50, 32), device)
        original_sample_noise = model.sample_noise

        def deterministic_noise(shape, dev):
            return fixed_noise.clone()

        model.sample_noise = deterministic_noise  # type: ignore[method-assign]

        with torch.no_grad():
            actions_no_rtc = model.predict_action_chunk(batch_no_rtc)
            model.enable_rtc = True
            actions_with_rtc = model.predict_action_chunk(batch_with_rtc)
            model.enable_rtc = False

        model.sample_noise = original_sample_noise  # type: ignore[method-assign]

        assert not torch.allclose(actions_no_rtc, actions_with_rtc, atol=1e-4)

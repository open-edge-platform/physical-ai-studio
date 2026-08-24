# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the XR1 flow-matching objective and sampler."""

from __future__ import annotations

import pytest
import torch
from physicalai.policies.xr1.model import WEIGHT_MAX, WEIGHT_MIN, XR1FlowModel

BATCH = 4
HORIZON = 6
ACTION_DIM = 8


@pytest.fixture
def flow() -> XR1FlowModel:
    """Return a flow model with a short sampling schedule.

    Returns:
        The flow model.
    """
    return XR1FlowModel(num_inference_steps=3, freq_excluded_dims=(5,))


class TestTimestepSampling:
    """Timestep priors."""

    @pytest.mark.parametrize("sampling", ["beta", "uniform", "logit_normal"])
    def test_in_unit_range(self, sampling: str) -> None:
        """Every supported prior stays inside [0, 1)."""
        flow = XR1FlowModel(flow_sampling=sampling)  # type: ignore[arg-type]
        timestep = flow.sample_timestep(64, torch.device("cpu"), torch.float32)

        assert timestep.shape == (64, 1, 1)
        assert float(timestep.min()) >= 0.0
        assert float(timestep.max()) < 1.0

    def test_beta_prior_concentrates_on_noisy_timesteps(self) -> None:
        """1 - Beta(1.5, 1.0) has mean 0.4, weighting the hard, noisy end of the path."""
        flow = XR1FlowModel(flow_sampling="beta")
        timestep = flow.sample_timestep(8192, torch.device("cpu"), torch.float32)

        assert float(timestep.mean()) == pytest.approx(0.4, abs=0.02)

    def test_rejects_unknown_sampling(self) -> None:
        """An unsupported prior fails loudly."""
        flow = XR1FlowModel()
        flow.flow_sampling = "gaussian"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unsupported flow_sampling"):
            flow.sample_timestep(1, torch.device("cpu"), torch.float32)


class TestSchedule:
    """Interpolation and targets."""

    def test_interpolation_endpoints(self, flow: XR1FlowModel) -> None:
        """t=0 returns the noise, t=1 the action."""
        noise = torch.randn(BATCH, HORIZON, ACTION_DIM)
        action = torch.randn(BATCH, HORIZON, ACTION_DIM)
        zeros = torch.zeros(BATCH, 1, 1)

        assert torch.allclose(flow.interpolate(noise, action, zeros), noise)
        assert torch.allclose(flow.interpolate(noise, action, zeros + 1), action)

    def test_velocity_target(self, flow: XR1FlowModel) -> None:
        """The target velocity transports noise onto the action."""
        noise = torch.randn(BATCH, HORIZON, ACTION_DIM)
        action = torch.randn(BATCH, HORIZON, ACTION_DIM)

        assert torch.allclose(flow.velocity_target(noise, action), action - noise)


class TestGenerate:
    """Euler integration."""

    def test_zero_field_is_identity(self, flow: XR1FlowModel) -> None:
        """With no velocity the sample never moves."""
        noise = torch.randn(BATCH, HORIZON, ACTION_DIM)

        assert torch.allclose(flow.generate(noise, lambda sample, _: torch.zeros_like(sample)), noise)

    def test_constant_field_integrates_to_one(self, flow: XR1FlowModel) -> None:
        """A unit field advances the sample by exactly one over the schedule."""
        noise = torch.zeros(1, HORIZON, ACTION_DIM)
        result = flow.generate(noise, lambda sample, _: torch.ones_like(sample))

        assert torch.allclose(result, torch.ones_like(result))

    def test_step_count_matches_config(self) -> None:
        """The sampler takes exactly num_inference_steps steps, starting at t=0."""
        flow = XR1FlowModel(num_inference_steps=7)
        timesteps: list[float] = []

        def record(sample: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
            timesteps.append(float(step[0, 0, 0]))
            return torch.zeros_like(sample)

        flow.generate(torch.zeros(1, 2, 2), record)

        assert len(timesteps) == 7
        assert timesteps[0] == pytest.approx(0.0)
        assert timesteps[-1] == pytest.approx(6 / 7)


class TestFlowLoss:
    """Weighted velocity MSE plus the frequency term."""

    def test_zero_loss_on_perfect_prediction(self, flow: XR1FlowModel) -> None:
        """An exact prediction scores zero on both terms."""
        target = torch.randn(BATCH, HORIZON, ACTION_DIM)
        loss_mse, loss_freq = flow.flow_loss(target.clone(), target, torch.ones_like(target))

        assert float(loss_mse) == pytest.approx(0.0, abs=1e-6)
        assert float(loss_freq) == pytest.approx(0.0, abs=1e-6)

    def test_empty_mask_returns_differentiable_zeros(self, flow: XR1FlowModel) -> None:
        """An unlucky batch must not break the optimizer step."""
        pred = torch.randn(BATCH, HORIZON, ACTION_DIM, requires_grad=True)
        target = torch.randn(BATCH, HORIZON, ACTION_DIM)

        loss_mse, loss_freq = flow.flow_loss(pred, target, torch.zeros_like(target))
        (loss_mse + loss_freq).backward()

        assert float(loss_mse) == 0.0
        assert float(loss_freq) == 0.0
        assert pred.grad is not None

    def test_padding_dimensions_are_ignored(self, flow: XR1FlowModel) -> None:
        """Error in a padded dimension must not affect the loss."""
        target = torch.zeros(BATCH, HORIZON, ACTION_DIM)
        pred = target.clone()
        pred[..., 6:] = 100.0
        mask = torch.zeros_like(target)
        mask[..., :6] = 1.0

        loss_mse, _ = flow.flow_loss(pred, target, mask)

        assert float(loss_mse) == pytest.approx(0.0, abs=1e-6)

    def test_weights_are_normalized_and_clamped(self, flow: XR1FlowModel) -> None:
        """Extreme weights cannot dominate a batch."""
        target = torch.zeros(2, HORIZON, ACTION_DIM)
        pred = torch.ones_like(target)
        mask = torch.ones_like(target)
        weight = torch.ones_like(target)
        weight[0] = 1e6

        loss_mse, _ = flow.flow_loss(pred, target, mask, weight)

        # Squared error is 1 everywhere, so the loss equals the mean clamped weight,
        # which is bounded by the clamp range rather than by 1e6.
        assert WEIGHT_MIN <= float(loss_mse) <= WEIGHT_MAX

    def test_frequency_term_can_be_disabled(self) -> None:
        """enable_freq=False skips the spectral term entirely."""
        flow = XR1FlowModel(enable_freq=False)
        target = torch.randn(BATCH, HORIZON, ACTION_DIM)

        _, loss_freq = flow.flow_loss(target + 1.0, target, torch.ones_like(target))

        assert float(loss_freq) == 0.0

    def test_excluded_dimensions_leave_frequency_untouched(self) -> None:
        """Gripper-like channels are meant to switch abruptly, so they are excluded."""
        flow = XR1FlowModel(freq_excluded_dims=(0,))
        target = torch.zeros(2, HORIZON, 2)
        pred = target.clone()
        pred[..., 0] = torch.randn(2, HORIZON)

        _, loss_freq = flow.flow_loss(pred, target, torch.ones_like(target))

        assert float(loss_freq) == pytest.approx(0.0, abs=1e-6)

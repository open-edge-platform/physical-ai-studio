# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Real-Time Chunking (RTC) mixins."""

from __future__ import annotations

import pytest
import torch

from physicalai.policies.mixins.rtc import RTC_CHECKPOINT_KEY, RTCModelMixin, RTCPolicyMixin


class _ModelStub(RTCModelMixin):
    """Minimal stand-in for a torch model exposing the RTC flag."""

    def __init__(self, chunk_size: int = 4) -> None:
        self.enable_rtc = False
        self._chunk_size = chunk_size


class _PolicyStub(RTCPolicyMixin):
    """Policy-like object mixing in the RTC API with an optional model."""

    def __init__(self, model: _ModelStub | None = None) -> None:
        self.model = model


class TestRtcEnabledProperty:
    """Tests for the ``rtc_enabled`` property getter/setter."""

    def test_default_disabled(self) -> None:
        """RTC is disabled by default."""
        policy = _PolicyStub(model=_ModelStub())
        assert policy.rtc_enabled is False

    def test_setter_forwards_to_existing_model(self) -> None:
        """Setting the property updates the underlying model flag."""
        model = _ModelStub()
        policy = _PolicyStub(model=model)

        policy.rtc_enabled = True

        assert model.enable_rtc is True
        assert policy.rtc_enabled is True

    def test_getter_reflects_model_flag(self) -> None:
        """The model flag is the live source once the model exists."""
        model = _ModelStub()
        policy = _PolicyStub(model=model)

        model.enable_rtc = True

        assert policy.rtc_enabled is True

    def test_setter_coerces_truthy_value_to_bool(self) -> None:
        """Truthy/falsy inputs are coerced to a plain bool on the model."""
        model = _ModelStub()
        policy = _PolicyStub(model=model)

        policy.rtc_enabled = 1  # type: ignore[assignment]

        assert model.enable_rtc is True


class TestLazyModel:
    """Tests for caching RTC state before the model is built."""

    def test_state_cached_without_model(self) -> None:
        """Setting RTC before a model exists caches the desired state."""
        policy = _PolicyStub(model=None)

        policy.rtc_enabled = True

        assert policy.rtc_enabled is True

    def test_sync_applies_cached_state(self) -> None:
        """Building the model then syncing applies the cached RTC state."""
        policy = _PolicyStub(model=None)
        policy.rtc_enabled = True

        policy.model = _ModelStub()
        policy._sync_rtc_to_model()

        assert policy.model.enable_rtc is True

    def test_sync_noop_without_model(self) -> None:
        """Syncing without a model does not raise."""
        policy = _PolicyStub(model=None)
        policy.rtc_enabled = True

        policy._sync_rtc_to_model()

        assert policy.rtc_enabled is True


class TestSupportsRtc:
    """Tests for the ``supports_rtc`` guard."""

    def test_enable_unsupported_raises(self) -> None:
        """Enabling RTC on a policy that does not support it raises."""

        class _NoRtcPolicy(_PolicyStub):
            supports_rtc = False

        policy = _NoRtcPolicy(model=_ModelStub())

        with pytest.raises(ValueError, match="does not support Real-Time Chunking"):
            policy.rtc_enabled = True

    def test_disable_unsupported_allowed(self) -> None:
        """Disabling RTC on an unsupported policy is always allowed."""

        class _NoRtcPolicy(_PolicyStub):
            supports_rtc = False

        policy = _NoRtcPolicy(model=_ModelStub())

        policy.rtc_enabled = False

        assert policy.rtc_enabled is False


class TestModelWithoutRtcSupport:
    """Tests for policies whose model does not implement the RTC mixin."""

    def test_state_stays_on_policy(self) -> None:
        """A non-RTC model is left untouched and the state stays cached."""

        class _PlainModel:
            pass

        model = _PlainModel()
        policy = _PolicyStub(model=model)  # type: ignore[arg-type]

        policy.rtc_enabled = True

        assert not hasattr(model, "enable_rtc")
        assert policy.rtc_enabled is True

    def test_sync_is_noop(self) -> None:
        """Syncing to a non-RTC model does not set the flag."""

        class _PlainModel:
            pass

        policy = _PolicyStub(model=None)
        policy.rtc_enabled = True
        policy.model = _PlainModel()  # type: ignore[assignment]

        policy._sync_rtc_to_model()

        assert not hasattr(policy.model, "enable_rtc")


class TestCheckpointPersistence:
    """Tests for saving/restoring the RTC toggle through checkpoints."""

    def test_save_writes_flag(self) -> None:
        """The enabled state is written to the checkpoint dict."""
        policy = _PolicyStub(model=_ModelStub())
        policy.rtc_enabled = True

        checkpoint: dict[str, object] = {}
        policy.on_save_checkpoint(checkpoint)

        assert checkpoint[RTC_CHECKPOINT_KEY] is True

    def test_load_restores_flag(self) -> None:
        """A saved state is restored onto the policy and its model."""
        model = _ModelStub()
        policy = _PolicyStub(model=model)

        policy.on_load_checkpoint({RTC_CHECKPOINT_KEY: True})

        assert policy.rtc_enabled is True
        assert model.enable_rtc is True

    def test_load_before_model_exists(self) -> None:
        """A restored state is applied once the model is built."""
        policy = _PolicyStub(model=None)

        policy.on_load_checkpoint({RTC_CHECKPOINT_KEY: True})
        model = _ModelStub()
        policy.model = model
        policy._sync_rtc_to_model()

        assert model.enable_rtc is True

    def test_load_legacy_checkpoint_is_noop(self) -> None:
        """Checkpoints without the RTC key leave the current state untouched."""
        model = _ModelStub()
        policy = _PolicyStub(model=model)
        policy.rtc_enabled = True

        policy.on_load_checkpoint({})

        assert policy.rtc_enabled is True

    def test_load_enabled_on_unsupported_policy(self) -> None:
        """Restoring an enabled flag onto an unsupported policy stays disabled."""

        class _NoRtcPolicy(_PolicyStub):
            supports_rtc = False

        policy = _NoRtcPolicy(model=_ModelStub())

        policy.on_load_checkpoint({RTC_CHECKPOINT_KEY: True})

        assert policy.rtc_enabled is False


class TestModelMixinFlag:
    """Tests for the model-side ``enable_rtc`` flag."""

    def test_default_disabled(self) -> None:
        """RTC is off unless a policy turns it on."""

        class _BareModel(RTCModelMixin):
            pass

        assert _BareModel().enable_rtc is False

    def test_instance_flag_is_independent(self) -> None:
        """Setting the flag on one instance does not affect another."""
        first, second = _ModelStub(), _ModelStub()

        first.enable_rtc = True

        assert second.enable_rtc is False


class TestModelPrefixWeights:
    """Tests for ``RTCModelMixin._compute_prefix_weights``."""

    def test_linear_schedule_values(self) -> None:
        """Weights ramp down from the delay index and clamp to [0, 1]."""
        model = _ModelStub(chunk_size=4)

        weights = model._compute_prefix_weights(
            inference_delay=torch.tensor(1.0),
            execution_horizon=torch.tensor(2.0),
        )

        assert weights.shape == (1, 4, 1)
        torch.testing.assert_close(weights.flatten(), torch.tensor([1.0, 0.5, 0.0, 0.0]))

    def test_exp_schedule_is_bounded_by_linear(self) -> None:
        """The exponential schedule decays no slower than the linear one."""
        model = _ModelStub(chunk_size=4)
        kwargs = {"inference_delay": torch.tensor(1.0), "execution_horizon": torch.tensor(2.0)}

        linear = model._compute_prefix_weights(**kwargs, prefix_attention_schedule="linear")
        exponential = model._compute_prefix_weights(**kwargs, prefix_attention_schedule="exp")

        assert torch.all(exponential <= linear)
        assert torch.all(exponential >= 0.0)

    def test_uses_host_chunk_size(self) -> None:
        """Output length follows the host model's chunk size."""
        weights = _ModelStub(chunk_size=7)._compute_prefix_weights(
            inference_delay=torch.tensor(0.0),
            execution_horizon=torch.tensor(3.0),
        )

        assert weights.shape == (1, 7, 1)


class TestModelRtcCorrect:
    """Tests for ``RTCModelMixin._rtc_correct``."""

    def test_zero_prefix_weights_leave_velocity_unchanged(self) -> None:
        """Actions past the guided prefix keep their predicted velocity."""
        v_t = torch.randn(1, 4, 2)

        corrected = RTCModelMixin._rtc_correct(
            x_t=torch.zeros(1, 4, 2),
            v_t=v_t,
            prev_chunk_left_over=torch.ones(1, 4, 2),
            prefix_weights=torch.zeros(1, 4, 1),
            time=0.5,
            max_guidance_weight=torch.tensor(5.0),
        )

        torch.testing.assert_close(corrected, v_t)

    @pytest.mark.parametrize(("max_guidance_weight", "expected"), [(0.5, -0.5), (10.0, -2.0)])
    def test_guidance_weight_is_capped(self, max_guidance_weight: float, expected: float) -> None:
        """The adaptive guidance weight is clamped by ``max_guidance_weight``."""
        corrected = RTCModelMixin._rtc_correct(
            x_t=torch.zeros(1, 2, 2),
            v_t=torch.zeros(1, 2, 2),
            prev_chunk_left_over=torch.ones(1, 2, 2),
            prefix_weights=torch.ones(1, 2, 1),
            time=0.5,
            max_guidance_weight=torch.tensor(max_guidance_weight),
        )

        torch.testing.assert_close(corrected, torch.full((1, 2, 2), expected))

    # The sampling loop starts at time=1.0 and stops before reaching 0.
    @pytest.mark.parametrize("time", [1.0, 0.25, 0.01])
    def test_finite_across_the_denoising_schedule(self, time: float) -> None:
        """The correction stays finite at the singular end of the schedule."""
        corrected = RTCModelMixin._rtc_correct(
            x_t=torch.randn(1, 4, 2),
            v_t=torch.randn(1, 4, 2),
            prev_chunk_left_over=torch.randn(1, 4, 2),
            prefix_weights=torch.ones(1, 4, 1),
            time=time,
            max_guidance_weight=torch.tensor(5.0),
        )

        assert torch.isfinite(corrected).all()

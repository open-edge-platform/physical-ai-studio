# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Real-Time Chunking (RTC) policy mixin."""

from __future__ import annotations

import pytest

from physicalai.policies.utils.rtc_mixin import RTCPolicyMixin


class _ModelStub:
    """Minimal stand-in for a torch model exposing the RTC flag."""

    def __init__(self) -> None:
        self.enable_rtc = False


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

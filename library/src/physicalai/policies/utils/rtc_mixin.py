# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) policy mixin.

Provides a public, policy-level API for enabling or disabling Real-Time
Chunking (RTC) inference. RTC improves temporal consistency between
consecutive action chunks by guiding the denoising process with the
unconsumed tail of the previous chunk.

The mixin owns the RTC on/off state (the single source of truth) and forwards
it to the underlying torch model's ``enable_rtc`` flag. Because policies may
build their model lazily (e.g. during Lightning ``setup``), the desired state
is cached on the policy and applied to the model via
:meth:`RTCPolicyMixin._sync_rtc_to_model` once the model exists.
"""

from __future__ import annotations


class RTCPolicyMixin:
    """Expose Real-Time Chunking (RTC) as a public policy capability.

    Mixed into a policy alongside the model. The policy is the authoritative
    owner of the RTC on/off state; the underlying model's ``enable_rtc`` flag is
    kept in sync with it.

    Attributes:
        supports_rtc: Whether this policy family supports RTC. Subclasses whose
            model has no RTC implementation should override this to ``False`` so
            that enabling RTC raises instead of silently doing nothing.
    """

    supports_rtc: bool = True

    _rtc_enabled: bool = False

    @property
    def rtc_enabled(self) -> bool:
        """Whether Real-Time Chunking is enabled for inference and export.

        The underlying model's ``enable_rtc`` flag is treated as the live value
        once the model exists; before that, the cached desired state is used.

        Returns:
            ``True`` if RTC is enabled, ``False`` otherwise.
        """
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "enable_rtc"):
            return bool(model.enable_rtc)
        return self._rtc_enabled

    @rtc_enabled.setter
    def rtc_enabled(self, value: bool) -> None:
        """Enable or disable Real-Time Chunking.

        Caches the desired state on the policy and, if the model has already
        been built, forwards it to the model's ``enable_rtc`` flag.

        Args:
            value: ``True`` to enable RTC, ``False`` to disable it.

        Raises:
            ValueError: If RTC is enabled on a policy that does not support it.
        """
        enabled = bool(value)
        if enabled and not self.supports_rtc:
            msg = f"{type(self).__name__} does not support Real-Time Chunking (RTC)."
            raise ValueError(msg)

        self._rtc_enabled = enabled

        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "enable_rtc"):
            model.enable_rtc = enabled

    def _sync_rtc_to_model(self) -> None:
        """Apply the cached RTC state to the underlying model.

        Called by the policy after the model has been (re)built so that a
        desired state set before model construction takes effect.
        """
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "enable_rtc"):
            model.enable_rtc = self._rtc_enabled

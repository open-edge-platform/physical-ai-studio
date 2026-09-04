# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real-Time Chunking (RTC) mixins.

Provides a public, policy-level API for enabling or disabling Real-Time
Chunking (RTC) inference, and the model-level math that implements it. RTC
improves temporal consistency between consecutive action chunks by guiding the
denoising process with the unconsumed tail of the previous chunk.

:class:`RTCPolicyMixin` owns the RTC on/off state (the single source of truth)
and forwards it to the underlying :class:`RTCModelMixin` model's ``enable_rtc``
flag. Because policies may build their model lazily (e.g. during Lightning
``setup``), the desired state is cached on the policy and applied to the model
via :meth:`RTCPolicyMixin._sync_rtc_to_model` once the model exists. The state
is also written to (and restored from) Lightning checkpoints under the
``rtc_enabled`` key.

:class:`RTCModelMixin` carries that ``enable_rtc`` flag on the flow-matching
model together with the guidance correction applied at each denoising step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from torch import Tensor

RTC_CHECKPOINT_KEY = "rtc_enabled"
"""Top-level checkpoint key holding the persisted RTC toggle."""


class RTCPolicyMixin:
    """Expose Real-Time Chunking (RTC) as a public policy capability.

    Mixed into a policy alongside the model. The policy is the authoritative
    owner of the RTC on/off state; the ``enable_rtc`` flag of the underlying
    :class:`RTCModelMixin` model is kept in sync with it.

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
        if isinstance(model, RTCModelMixin):
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
        if isinstance(model, RTCModelMixin):
            model.enable_rtc = enabled

    def _sync_rtc_to_model(self) -> None:
        """Apply the cached RTC state to the underlying model.

        Called by the policy after the model has been (re)built so that a
        desired state set before model construction takes effect.
        """
        model = getattr(self, "model", None)
        if isinstance(model, RTCModelMixin):
            model.enable_rtc = self._rtc_enabled

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist the RTC toggle alongside the Lightning checkpoint.

        Args:
            checkpoint: Checkpoint dictionary being written.
        """
        hook = getattr(super(), "on_save_checkpoint", None)
        if hook is not None:
            hook(checkpoint)
        checkpoint[RTC_CHECKPOINT_KEY] = self.rtc_enabled

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the RTC toggle from a Lightning checkpoint.

        Checkpoints written before RTC was persisted simply leave the current
        state untouched.

        Args:
            checkpoint: Checkpoint dictionary being loaded.
        """
        hook = getattr(super(), "on_load_checkpoint", None)
        if hook is not None:
            hook(checkpoint)
        stored = checkpoint.get(RTC_CHECKPOINT_KEY)
        if stored is None:
            return
        self.rtc_enabled = bool(stored) and self.supports_rtc


class RTCModelMixin:
    """Give a flow-matching ``nn.Module`` the Real-Time Chunking inference math.

    Mixed into a model alongside ``nn.Module``. Holds the ``enable_rtc`` toggle
    that :class:`RTCPolicyMixin` writes to, and implements the per-step guidance
    correction that pins the start of the new chunk to the unconsumed tail of
    the previous one.

    The host model must expose a ``_chunk_size`` attribute (the number of
    actions per predicted chunk).

    Attributes:
        enable_rtc: Whether RTC inputs are consumed at inference time. Owned by
            the policy; see :class:`RTCPolicyMixin`.
    """

    enable_rtc: bool = False

    # Declared for type checkers only; provided by the host model.
    _chunk_size: int
    _max_action_dim: int

    def _pad_prev_chunk(self, prev_chunk: Tensor | None) -> Tensor | None:
        """Pad the previous action chunk to the model's action dimension.

        The chunk arrives in the dataset's action dimension, while the denoised
        state it guides lives in ``_max_action_dim``.

        Args:
            prev_chunk: Unconsumed tail of the previous chunk, or ``None``.

        Returns:
            The chunk zero-padded along its last dimension, or ``None``.
        """
        if prev_chunk is None:
            return None
        padding = self._max_action_dim - prev_chunk.shape[-1]
        if padding <= 0:
            return prev_chunk
        return torch.nn.functional.pad(prev_chunk, (0, padding))

    def _compute_prefix_weights(
        self,
        inference_delay: Tensor,
        execution_horizon: Tensor,
        prefix_attention_schedule: Literal["linear", "exp"] = "linear",
    ) -> Tensor:
        """Compute prefix attention weights inside the graph.

        Args:
            inference_delay: Scalar tensor — the dynamic latency estimate.
            execution_horizon: Scalar tensor — number of fresh actions per chunk.
            prefix_attention_schedule: Schedule type for prefix attention weights ("linear" or "exp").

        Returns:
            ``(1, chunk_size, 1)`` weight tensor.
        """
        chunk_size = self._chunk_size
        end = execution_horizon.float()
        start = torch.minimum(inference_delay.float(), end)

        idx = torch.arange(chunk_size, dtype=torch.float32, device=inference_delay.device)
        denom = end - start + 1.0
        weights = (end - idx) / denom
        weights = torch.clamp(weights, min=0.0, max=1.0)

        if prefix_attention_schedule == "exp":
            weights = weights * (torch.exp(weights) - 1.0) / (math.e - 1.0)
        # "linear" → no-op

        return weights.unsqueeze(0).unsqueeze(-1)  # (1, chunk_size, 1)

    @staticmethod
    def _rtc_correct(
        x_t: Tensor,
        v_t: Tensor,
        prev_chunk_left_over: Tensor,
        prefix_weights: Tensor,
        time: float,
        max_guidance_weight: Tensor,
    ) -> Tensor:
        """Apply RTC guidance correction to velocity prediction.

        Uses direct error (not autograd.grad) for OV traceability.

        Returns:
            Corrected velocity tensor.
        """
        tau = 1.0 - time

        # Predicted clean actions at t=0
        x1_t = x_t - time * v_t

        # Weighted error between previous chunk and prediction
        err = (prev_chunk_left_over - x1_t) * prefix_weights
        correction = err

        # Adaptive guidance weight
        max_gw = max_guidance_weight.to(dtype=torch.float32)
        tau_t = torch.as_tensor(tau, dtype=max_gw.dtype, device=max_gw.device)
        squared_one_minus_tau = (1.0 - tau_t) ** 2
        inv_r2 = (squared_one_minus_tau + tau_t**2) / squared_one_minus_tau

        # Manual nan_to_num — torch.nan_to_num not supported by OV
        c_raw = (1.0 - tau_t) / tau_t
        c = torch.where(torch.isinf(c_raw), max_gw, c_raw)

        guidance_weight_raw = c * inv_r2
        guidance_weight = torch.where(torch.isinf(guidance_weight_raw), max_gw, guidance_weight_raw)
        guidance_weight = torch.minimum(guidance_weight, max_gw)

        return v_t - guidance_weight * correction

    def _validate_rtc_inputs(
        self,
        inference_delay: float | Tensor,
        execution_horizon: float | Tensor,
        max_guidance_weight: float | Tensor,
    ) -> None:
        """Validate the RTC control values parsed from an inference batch.

        Values may arrive as Python numbers or as scalar tensors carried in the
        batch. Validation is skipped while tracing or exporting, where the values
        are graph placeholders rather than concrete numbers.

        Args:
            inference_delay: Number of actions expected to be consumed while the
                new chunk is still being computed.
            execution_horizon: Number of fresh actions taken from the new chunk.
            max_guidance_weight: Upper bound on the RTC guidance weight.

        Raises:
            ValueError: If the timing values do not satisfy
                ``0 <= inference_delay <= execution_horizon <= chunk_size``, if
                ``execution_horizon`` is not positive, or if
                ``max_guidance_weight`` is negative.
        """
        if torch.jit.is_tracing() or torch.onnx.is_in_onnx_export():
            return

        delay, horizon, guidance = (
            float(value.flatten()[0]) if isinstance(value, torch.Tensor) else float(value)
            for value in (inference_delay, execution_horizon, max_guidance_weight)
        )

        if horizon <= 0:
            msg = f"RTC execution_horizon must be positive, got {horizon}."
            raise ValueError(msg)

        if not 0 <= delay <= horizon <= self._chunk_size:
            msg = (
                "RTC timing values must satisfy 0 <= inference_delay <= execution_horizon <= chunk_size, "
                f"got inference_delay={delay}, execution_horizon={horizon}, chunk_size={self._chunk_size}."
            )
            raise ValueError(msg)

        if guidance < 0:
            msg = f"RTC max_guidance_weight must be non-negative, got {guidance}."
            raise ValueError(msg)

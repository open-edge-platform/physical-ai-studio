# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared SnapFlow policy and config surface.

SnapFlow is implemented once in :mod:`physicalai.policies.mixins.snapflow` and
mixed into both Pi05 and SmolVLA. These tests cover the shared surface; the
per-policy hooks are covered by asserting each policy binds the shared
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from physicalai.config import Config
from physicalai.policies import Pi05, SmolVLA
from physicalai.policies.mixins import SnapFlowConfigMixin, SnapFlowPolicyMixin
from physicalai.policies.pi05 import Pi05Config
from physicalai.policies.smolvla import SmolVLAConfig

_SNAPFLOW_FIELDS = (
    "snapflow_enabled",
    "snapflow_alpha",
    "snapflow_lambda",
    "snapflow_num_inference_steps",
)


# ============================================================================ #
# Config surface                                                               #
# ============================================================================ #


class TestSnapFlowConfigMixin:
    """Both policy configs inherit the same SnapFlow flags and validation."""

    @pytest.mark.parametrize("config_cls", [Pi05Config, SmolVLAConfig])
    def test_defaults(self, config_cls: type[Config]) -> None:
        config = config_cls()
        assert config.snapflow_enabled is False
        assert config.snapflow_alpha == 0.5
        assert config.snapflow_lambda == 0.1
        assert config.snapflow_num_inference_steps == 1

    @pytest.mark.parametrize("config_cls", [Pi05Config, SmolVLAConfig])
    def test_fields_survive_dict_round_trip(self, config_cls: type[Config]) -> None:
        """Checkpoint hparams are plain dicts, so the flags must serialize."""
        config = config_cls(snapflow_enabled=True, snapflow_alpha=0.25, snapflow_num_inference_steps=2)
        as_dict = config.to_dict()

        assert all(field in as_dict for field in _SNAPFLOW_FIELDS)
        assert config_cls.from_dict(as_dict) == config

    @pytest.mark.parametrize("config_cls", [Pi05Config, SmolVLAConfig])
    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_rejects_alpha_outside_unit_interval(self, config_cls: type[Config], alpha: float) -> None:
        with pytest.raises(ValueError, match=r"snapflow_alpha must be in \[0, 1\]"):
            config_cls(snapflow_alpha=alpha)

    @pytest.mark.parametrize("config_cls", [Pi05Config, SmolVLAConfig])
    def test_rejects_zero_inference_steps(self, config_cls: type[Config]) -> None:
        with pytest.raises(ValueError, match="snapflow_num_inference_steps must be >= 1"):
            config_cls(snapflow_num_inference_steps=0)

    @pytest.mark.parametrize("config_cls", [Pi05Config, SmolVLAConfig])
    def test_accepts_unit_interval_bounds(self, config_cls: type[Config]) -> None:
        assert config_cls(snapflow_alpha=0.0).snapflow_alpha == 0.0
        assert config_cls(snapflow_alpha=1.0).snapflow_alpha == 1.0


# ============================================================================ #
# Policy mixin                                                                 #
# ============================================================================ #


@dataclass(frozen=True)
class _StubConfig(SnapFlowConfigMixin, Config):
    """Minimal frozen config carrying the flags the mixin mutates."""

    train_expert_only: bool = False


class _StubInner(torch.nn.Module):
    """Stand-in for Pi05Model / VLAFlowMatching."""

    def __init__(self) -> None:
        super().__init__()
        self._snapflow_enabled = False
        self._snapflow_alpha = 0.5
        self._snapflow_lambda = 1.0
        self._snapflow_num_inference_steps = 10


class _StubPolicy(SnapFlowPolicyMixin):
    """Minimal host implementing the two required capabilities."""

    def __init__(self, *, model: _StubInner | None = None) -> None:
        self.config: Any = _StubConfig()
        self.model = model
        self.frozen = False
        self.hparams_synced = False

    @property
    def inner_model(self) -> _StubInner:
        if self.model is None:
            msg = "inner_model accessed before the model was initialized (setup() has not run yet)."
            raise RuntimeError(msg)
        return self.model

    def freeze_vlm(self) -> None:
        object.__setattr__(self.config, "train_expert_only", True)
        self.frozen = True

    def _set_hparam_keys(self) -> None:
        self.hparams_synced = True


class TestSnapFlowPolicyMixin:
    """Behaviour of the shared ``enable_snapflow`` entry point."""

    def test_sets_model_flags(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        policy.enable_snapflow(alpha=0.4, lambda_=0.2, num_inference_steps=2)

        inner = policy.model
        assert inner is not None
        assert inner._snapflow_enabled is True
        assert inner._snapflow_alpha == 0.4
        assert inner._snapflow_lambda == 0.2
        assert inner._snapflow_num_inference_steps == 2

    def test_mutates_frozen_config_so_checkpoints_reload_as_snapflow(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        policy.enable_snapflow(alpha=0.4, lambda_=0.2, num_inference_steps=2)

        assert policy.config.snapflow_enabled is True
        assert policy.config.snapflow_alpha == 0.4
        assert policy.config.snapflow_lambda == 0.2
        assert policy.config.snapflow_num_inference_steps == 2
        assert policy.config.train_expert_only is True

    def test_freezes_backbone_and_syncs_hparams(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        policy.enable_snapflow()

        assert policy.frozen is True
        assert policy.hparams_synced is True

    def test_paper_defaults(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        policy.enable_snapflow()

        assert policy.config.snapflow_alpha == 0.5
        assert policy.config.snapflow_lambda == 0.1
        assert policy.config.snapflow_num_inference_steps == 1

    def test_raises_before_model_is_initialized(self) -> None:
        policy = _StubPolicy(model=None)

        with pytest.raises(RuntimeError, match="before the model was initialized"):
            policy.enable_snapflow()

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_rejects_alpha_outside_unit_interval(self, alpha: float) -> None:
        policy = _StubPolicy(model=_StubInner())

        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            policy.enable_snapflow(alpha=alpha)

    def test_rejects_zero_inference_steps(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        with pytest.raises(ValueError, match="num_inference_steps must be >= 1"):
            policy.enable_snapflow(num_inference_steps=0)

    def test_invalid_arguments_leave_state_untouched(self) -> None:
        policy = _StubPolicy(model=_StubInner())

        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            policy.enable_snapflow(alpha=2.0)

        assert policy.config.snapflow_enabled is False
        assert policy.frozen is False

    def test_hooks_are_required(self) -> None:
        class _Incomplete(SnapFlowPolicyMixin):
            pass

        with pytest.raises(NotImplementedError, match="inner_model property"):
            _ = _Incomplete().inner_model
        with pytest.raises(NotImplementedError, match="freeze_vlm"):
            _Incomplete().freeze_vlm()


class TestPoliciesShareTheMixin:
    """Pi05 and SmolVLA must not carry divergent copies of the logic."""

    @pytest.mark.parametrize("policy_cls", [Pi05, SmolVLA])
    def test_uses_shared_enable_snapflow(self, policy_cls: type[SnapFlowPolicyMixin]) -> None:
        assert policy_cls.enable_snapflow is SnapFlowPolicyMixin.enable_snapflow

    @pytest.mark.parametrize("policy_cls", [Pi05, SmolVLA])
    def test_implements_required_capabilities(self, policy_cls: type[SnapFlowPolicyMixin]) -> None:
        assert policy_cls.inner_model is not SnapFlowPolicyMixin.inner_model
        assert policy_cls.freeze_vlm is not SnapFlowPolicyMixin.freeze_vlm

    @pytest.mark.parametrize("policy_cls", [Pi05, SmolVLA])
    def test_guards_against_uninitialized_model(self, policy_cls: type[SnapFlowPolicyMixin]) -> None:
        # Bypass __init__ so `model` is never assigned, mimicking a policy whose
        # setup() has not run yet.
        policy = object.__new__(policy_cls)
        policy.model = None

        with pytest.raises(RuntimeError, match="before the model was initialized"):
            policy.enable_snapflow()
        with pytest.raises(RuntimeError, match="before the model was initialized"):
            _ = policy.inner_model

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
from physicalai.policies.mixins import SnapFlowConfigMixin, SnapFlowModelMixin, SnapFlowPolicyMixin
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




# ============================================================================ #
# Mixed FM/consistency loss                                                    #
# ============================================================================ #


class _StubFlowModel(SnapFlowModelMixin):
    """Minimal host for ``snapflow_mixed_loss``: records which indices each
    velocity-prediction call touches, so tests can assert on the FM/CD split
    without a real flow-matching model."""

    def __init__(self, *, alpha: float, lambda_: float = 0.1) -> None:
        self.init_snapflow_state(enabled=True, alpha=alpha, lambda_=lambda_, num_inference_steps=1)
        self.predict_velocity_call_shapes: list[int] = []
        self.predict_velocity_call_markers: list[torch.Tensor] = []

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        _timestep: torch.Tensor,
        _target_time: torch.Tensor,
        prefix_embs: torch.Tensor,
        _prefix_pad_masks: torch.Tensor,
        _prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        self.predict_velocity_call_shapes.append(x_t.shape[0])
        # prefix_embs[:, 0, 0] carries a per-row identity marker in these
        # tests (see _mixed_loss_inputs), which lets tests recover exactly
        # which original rows landed in this call's branch.
        self.predict_velocity_call_markers.append(prefix_embs[:, 0, 0].clone())
        # A predictable function of the conditioning, so FM and CD losses are
        # both well-defined and distinguishable in tests.
        return x_t + prefix_embs.mean(dim=(1, 2), keepdim=True).expand_as(x_t)


def _mixed_loss_inputs(bsize: int, chunk: int = 4, dim: int = 3) -> dict[str, torch.Tensor]:
    """Build minimal-but-real tensors for ``snapflow_mixed_loss``.

    ``prefix_embs[:, 0, 0]`` is set to ``arange(bsize)`` so tests can recover
    exactly which original rows a given ``predict_velocity`` call touched.
    """
    prefix_embs = torch.randn(bsize, 2, dim)
    prefix_embs[:, 0, 0] = torch.arange(bsize, dtype=prefix_embs.dtype)
    return {
        "u_t": torch.randn(bsize, chunk, dim),
        "x_t": torch.randn(bsize, chunk, dim),
        "time": torch.rand(bsize),
        "actions": torch.randn(bsize, chunk, dim),
        "prefix_embs": prefix_embs,
        "prefix_pad_masks": torch.ones(bsize, 2, dtype=torch.bool),
        "prefix_att_masks": torch.ones(bsize, 2, dtype=torch.bool),
    }


class TestSnapFlowMixedLoss:
    """The FM/CD split must be a static-size, compile-friendly partition.

    See docs/explanation/policy/snapflow-two-phase-distillation.md ("Proposed
    design: compile-friendly SnapFlow loss") for the rationale: a fixed-size
    ``torch.randperm`` split instead of a per-sample Bernoulli mask, so
    ``torch.compile`` sees a static shape per branch instead of a
    data-dependent one.
    """

    @pytest.mark.parametrize(
        ("alpha", "bsize", "expected_n_fm"),
        [
            (0.5, 8, 4),
            (0.5, 7, 4),  # round(3.5) == 4 (banker's rounding lands here for .5)
            (0.25, 8, 2),
            (0.3, 7, 2),
            (0.0, 8, 0),
            (1.0, 8, 8),
        ],
    )
    def test_split_size_is_deterministic_given_alpha_and_bsize(
        self, alpha: float, bsize: int, expected_n_fm: int
    ) -> None:
        """The split size must be an exact function of (alpha, bsize), not a
        random count, so torch.compile only ever sees one shape per config."""
        model = _StubFlowModel(alpha=alpha)
        inputs = _mixed_loss_inputs(bsize)

        for _ in range(5):
            model.predict_velocity_call_shapes.clear()
            model.snapflow_mixed_loss(
                sample_noise=model.sample_noise,
                predict_velocity=model.predict_velocity,
                **inputs,
            )
            shapes = model.predict_velocity_call_shapes
            # Every call must be one of exactly two sizes: the (constant) FM
            # branch size or the (constant) CD branch size. No other shape
            # should ever appear, across repeated calls with fresh randomness.
            assert all(s in (expected_n_fm, bsize - expected_n_fm) for s in shapes)
            # Exactly one FM call (if the branch runs) and exactly three CD
            # calls (v_1, v_half, v_pred, if that branch runs). Counted by
            # call count rather than by size, because alpha=0.5 on an even
            # batch makes the two branch sizes coincide.
            expected_n_calls = (1 if expected_n_fm > 0 else 0) + (3 if expected_n_fm < bsize else 0)
            assert len(shapes) == expected_n_calls

    def test_pure_fm_skips_the_cd_branch_entirely(self) -> None:
        """alpha=1.0: no CD velocity calls, and every sample gets a real FM loss."""
        model = _StubFlowModel(alpha=1.0)
        inputs = _mixed_loss_inputs(bsize=6)

        losses = model.snapflow_mixed_loss(
            sample_noise=model.sample_noise, predict_velocity=model.predict_velocity, **inputs
        )

        assert model.predict_velocity_call_shapes == [6]  # only the FM branch call
        assert not torch.any(losses == 0)

    def test_pure_cd_skips_the_fm_branch_entirely(self) -> None:
        """alpha=0.0: no FM velocity calls, every sample goes through CD."""
        model = _StubFlowModel(alpha=0.0)
        inputs = _mixed_loss_inputs(bsize=6)

        losses = model.snapflow_mixed_loss(
            sample_noise=model.sample_noise, predict_velocity=model.predict_velocity, **inputs
        )

        assert model.predict_velocity_call_shapes == [6, 6, 6]  # v_1, v_half, v_pred
        assert not torch.any(losses == 0)

    def test_fm_and_cd_index_sets_partition_the_batch(self) -> None:
        """The FM and CD branches must not overlap and must cover every row,
        and every row of `losses` must be written by exactly one branch."""
        bsize = 10
        model = _StubFlowModel(alpha=0.5)
        inputs = _mixed_loss_inputs(bsize)

        losses = model.snapflow_mixed_loss(
            sample_noise=model.sample_noise, predict_velocity=model.predict_velocity, **inputs
        )

        fm_markers = model.predict_velocity_call_markers[0]
        cd_markers = model.predict_velocity_call_markers[1]  # v_1's call; identical across v_1/v_half/v_pred
        all_indices = torch.cat([fm_markers, cd_markers]).sort().values
        assert torch.equal(all_indices, torch.arange(bsize, dtype=all_indices.dtype))
        assert losses.shape == inputs["actions"].shape
        assert not torch.any(losses == 0)

    def test_split_membership_matches_alpha_marginal_across_many_trials(self) -> None:
        """A single batch has a static split *size*, but the point of using
        ``torch.randperm`` (over a fixed-order slice) is that which physical
        row lands in which branch still varies uniformly, so every sample
        keeps the same marginal probability ``alpha`` of landing in FM that
        the Bernoulli mask it replaces gave it."""
        alpha = 0.3
        bsize = 20
        model = _StubFlowModel(alpha=alpha)
        fm_hit_counts = torch.zeros(bsize)
        n_trials = 300

        for _ in range(n_trials):
            inputs = _mixed_loss_inputs(bsize)
            model.predict_velocity_call_markers.clear()
            model.snapflow_mixed_loss(
                sample_noise=model.sample_noise, predict_velocity=model.predict_velocity, **inputs
            )
            fm_markers = model.predict_velocity_call_markers[0]
            fm_hit_counts[fm_markers.long()] += 1

        empirical_rate = fm_hit_counts / n_trials
        assert abs(empirical_rate.mean().item() - alpha) < 0.02
        # No row should be systematically favored or excluded by the
        # permutation -- every row's individual empirical rate should also
        # sit close to alpha, not just the batch-wide average.
        assert (empirical_rate - alpha).abs().max().item() < 0.15

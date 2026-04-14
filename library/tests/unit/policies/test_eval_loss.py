# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for eval-loss validation step behavior.

Verifies that val/loss computation does not activate dropout or other
training-only behaviors that would bias the validation signal.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from physicalai.data.observation import Observation
from physicalai.policies.base.policy import Policy


# ---------------------------------------------------------------------------
# Minimal concrete policy with dropout for testing
# ---------------------------------------------------------------------------


class _DropoutPolicy(Policy):
    """Tiny policy with a dropout layer to test train-vs-eval loss behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Dropout(p=0.5)  # aggressive dropout to amplify the effect

        # Fixed weights so forward is deterministic (apart from dropout)
        nn.init.ones_(self.linear.weight)

    # -- required by base Policy --

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        if self.training:
            return self._loss(batch)
        # In eval mode, return a dummy action (base Policy.forward contract)
        return torch.zeros(1)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        return torch.zeros(1)

    def _loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch.state
        x = self.dropout(self.linear(x))
        loss = x.mean()
        return loss, {"loss": loss.item()}

    # -- override compute_val_loss to NOT toggle train mode --

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        return self._loss(batch)


class _NoOverridePolicy(_DropoutPolicy):
    """Same as _DropoutPolicy, but does NOT override compute_val_loss.

    Uses the base Policy default which raises NotImplementedError.
    """

    compute_val_loss = Policy.compute_val_loss  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_batch() -> Observation:
    torch.manual_seed(0)
    return Observation(state=torch.randn(2, 4))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalLossTrainModeLeak:
    """Verify that validation_step does not leak training-mode behavior."""

    def test_compute_val_loss_override_keeps_eval_mode(self) -> None:
        """Policy with compute_val_loss override stays in eval mode throughout."""
        policy = _DropoutPolicy()
        policy.eval()

        batch = _make_batch()
        # Record submodule modes during compute_val_loss
        modes_during_loss: list[bool] = []
        orig_forward = policy.linear.forward

        def tracking_forward(x: torch.Tensor) -> torch.Tensor:
            modes_during_loss.append(policy.training)
            return orig_forward(x)

        policy.linear.forward = tracking_forward  # type: ignore[method-assign]
        policy.compute_val_loss(batch)

        assert not any(modes_during_loss), "compute_val_loss should NOT set training=True"
        assert not policy.training, "Policy should still be in eval mode after compute_val_loss"

    def test_default_compute_val_loss_raises(self) -> None:
        """Base Policy.compute_val_loss (no override) raises NotImplementedError."""
        policy = _NoOverridePolicy()
        policy.eval()

        batch = _make_batch()

        with pytest.raises(NotImplementedError):
            policy.compute_val_loss(batch)

    def test_dropout_variance_without_override_raises(self) -> None:
        """Policy without compute_val_loss override raises NotImplementedError."""
        policy = _NoOverridePolicy()
        policy.eval()

        batch = _make_batch()

        with pytest.raises(NotImplementedError):
            policy.compute_val_loss(batch)

    def test_eval_mode_loss_is_deterministic(self) -> None:
        """Loss computed via overridden compute_val_loss (eval mode) is deterministic."""
        policy = _DropoutPolicy()
        policy.eval()

        batch = _make_batch()

        eval_mode_losses = []
        for _ in range(20):
            loss, _ = policy.compute_val_loss(batch)
            eval_mode_losses.append(loss.item())

        assert len(set(eval_mode_losses)) == 1, "Eval-mode losses should be identical (no dropout)"

    def test_validation_step_uses_compute_val_loss(self) -> None:
        """validation_step delegates to compute_val_loss for Observation batches."""
        policy = _DropoutPolicy()
        policy.eval()

        batch = _make_batch()

        # Mock self.log to avoid Lightning errors
        logged: dict[str, float] = {}
        policy.log = lambda name, value, **kwargs: logged.update({name: value})  # type: ignore[assignment]

        loss = policy.validation_step(batch, batch_idx=0)

        assert "val/loss" in logged
        assert not policy.training, "Should remain in eval mode"

        # Loss should match a direct compute_val_loss call
        expected_loss, _ = policy.compute_val_loss(batch)
        assert loss.item() == expected_loss.item()

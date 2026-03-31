# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for learning rate schedulers."""

from __future__ import annotations

import math

import pytest
import torch

from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler


@pytest.fixture
def optimizer():
    """Simple optimizer for testing."""
    model = torch.nn.Linear(2, 2)
    return torch.optim.AdamW(model.parameters(), lr=2.5e-5)


class TestCosineDecayWithWarmupScheduler:
    """Tests for cosine_decay_with_warmup_scheduler."""

    PEAK_LR = 2.5e-5
    DECAY_LR = 2.5e-6
    WARMUP_STEPS = 100
    DECAY_STEPS = 1000

    def _make_scheduler(self, optimizer):
        return cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.PEAK_LR,
            decay_lr=self.DECAY_LR,
            num_warmup_steps=self.WARMUP_STEPS,
            num_decay_steps=self.DECAY_STEPS,
        )

    def test_initial_lr(self, optimizer):
        """LR at step 0 should be peak_lr / (warmup_steps + 1)."""
        scheduler = self._make_scheduler(optimizer)
        expected = self.PEAK_LR / (self.WARMUP_STEPS + 1)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-5)
        # step 0 shouldn't change the multiplier (already applied)
        scheduler.step()

    def test_warmup_increases_lr(self, optimizer):
        """LR should increase during warmup."""
        scheduler = self._make_scheduler(optimizer)
        lrs = []
        for _ in range(self.WARMUP_STEPS):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        # LR should be monotonically increasing
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1]

    def test_end_of_warmup_reaches_peak(self, optimizer):
        """LR at last warmup step should be approximately peak_lr."""
        scheduler = self._make_scheduler(optimizer)
        for _ in range(self.WARMUP_STEPS - 1):
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(self.PEAK_LR, rel=2e-2)

    def test_cosine_decay_decreases_lr(self, optimizer):
        """LR should decrease after warmup during cosine decay."""
        scheduler = self._make_scheduler(optimizer)
        # Advance past warmup
        for _ in range(self.WARMUP_STEPS):
            scheduler.step()
        lr_after_warmup = optimizer.param_groups[0]["lr"]

        # Advance into decay
        for _ in range(self.DECAY_STEPS // 2):
            scheduler.step()
        lr_mid_decay = optimizer.param_groups[0]["lr"]

        assert lr_mid_decay < lr_after_warmup

    def test_end_of_decay_reaches_floor(self, optimizer):
        """LR at end of decay should be approximately decay_lr."""
        scheduler = self._make_scheduler(optimizer)
        for _ in range(self.DECAY_STEPS):
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(self.DECAY_LR, rel=1e-2)

    def test_lr_stays_at_floor_after_decay(self, optimizer):
        """LR should not drop below decay_lr after decay is complete."""
        scheduler = self._make_scheduler(optimizer)
        for _ in range(self.DECAY_STEPS + 500):
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(self.DECAY_LR, rel=1e-2)

    def test_matches_lerobot_schedule(self, optimizer):
        """Verify schedule matches LeRobot's CosineDecayWithWarmupScheduler."""
        scheduler = self._make_scheduler(optimizer)
        alpha = self.DECAY_LR / self.PEAK_LR

        prev_step = 0
        for step in [0, 10, 50, 99, 100, 250, 500, 750, 1000, 1500]:
            # Advance scheduler to the target step
            for _ in range(step - prev_step):
                scheduler.step()
            prev_step = step

            # Compute expected multiplier (from lerobot's lr_lambda)
            if step < self.WARMUP_STEPS:
                if step <= 0:
                    expected_mult = 1.0 / (self.WARMUP_STEPS + 1)
                else:
                    frac = 1.0 - step / self.WARMUP_STEPS
                    expected_mult = (1.0 / (self.WARMUP_STEPS + 1) - 1.0) * frac + 1.0
            else:
                clamped = min(step, self.DECAY_STEPS)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * clamped / self.DECAY_STEPS))
                expected_mult = (1.0 - alpha) * cosine_decay + alpha

            expected_lr = self.PEAK_LR * expected_mult
            actual_lr = optimizer.param_groups[0]["lr"]
            assert actual_lr == pytest.approx(expected_lr, rel=1e-5), (
                f"Mismatch at step {step}: expected {expected_lr}, got {actual_lr}"
            )

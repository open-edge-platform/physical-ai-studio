# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy wrappers."""

from __future__ import annotations

import pytest


class TestLeRobotPolicyValidation:
    """Tests for LeRobot policy validation functionality."""

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_lerobot_policy_has_validation_step(self):
        """Test LeRobot policy validation_step method exists."""
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import LeRobotPolicy

        # This will be lazily initialized in setup()
        policy = LeRobotPolicy(
            policy_name="diffusion",
            learning_rate=1e-4,
            num_steps=10,
        )

        # Note: Without calling setup(), the policy won't be fully initialized
        # But we can at least verify the method signature exists
        assert hasattr(policy, "validation_step")
        assert callable(policy.validation_step)

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_lerobot_policy_has_test_step(self):
        """Test LeRobot policy test_step method exists."""
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import LeRobotPolicy

        policy = LeRobotPolicy(
            policy_name="diffusion",
            learning_rate=1e-4,
            num_steps=10,
        )

        assert hasattr(policy, "test_step")
        assert callable(policy.test_step)

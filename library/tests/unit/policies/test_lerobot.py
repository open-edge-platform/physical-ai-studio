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


class TestGrootPolicy:
    """Tests for Groot policy wrapper."""

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_groot_import(self):
        """Test Groot can be imported."""
        from getiaction.policies.lerobot import Groot

        assert Groot is not None

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_groot_instantiation(self):
        """Test Groot can be instantiated with default parameters."""
        from getiaction.policies.lerobot import Groot

        # This will be lazily initialized in setup()
        policy = Groot(
            chunk_size=50,
            n_action_steps=50,
            learning_rate=1e-4,
        )

        assert hasattr(policy, "validation_step")
        assert hasattr(policy, "test_step")
        assert hasattr(policy, "training_step")
        assert hasattr(policy, "select_action")
        assert hasattr(policy, "reset")

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_groot_config_storage(self):
        """Test Groot stores config kwargs correctly."""
        from getiaction.policies.lerobot import Groot

        policy = Groot(
            chunk_size=100,
            n_action_steps=50,
            tune_llm=True,
            lora_rank=16,
        )

        assert policy._config_kwargs["chunk_size"] == 100
        assert policy._config_kwargs["n_action_steps"] == 50
        assert policy._config_kwargs["tune_llm"] is True
        assert policy._config_kwargs["lora_rank"] == 16


class TestUniversalPolicies:
    """Tests for universal policy wrapper with various policy types."""

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    @pytest.mark.parametrize(
        "policy_name",
        ["act", "diffusion", "vqbet", "tdmpc", "sac", "pi0", "pi05", "pi0fast", "smolvla", "groot"],
    )
    def test_universal_wrapper_instantiation(self, policy_name: str):
        """Test LeRobotPolicy can be instantiated with all supported policy types."""
        from getiaction.policies.lerobot import LeRobotPolicy

        # This will be lazily initialized in setup()
        policy = LeRobotPolicy(
            policy_name=policy_name,
            learning_rate=1e-4,
        )

        assert policy.policy_name == policy_name
        assert hasattr(policy, "validation_step")
        assert hasattr(policy, "test_step")

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_unsupported_policy_raises(self):
        """Test LeRobotPolicy raises error for unsupported policy."""
        from getiaction.policies.lerobot import LeRobotPolicy

        with pytest.raises(ValueError, match="not supported"):
            LeRobotPolicy(policy_name="unsupported_policy")


class TestConvenienceClasses:
    """Tests for convenience policy classes."""

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_convenience_class_groot(self):
        """Test Groot explicit wrapper is distinct from universal wrapper."""
        from getiaction.policies.lerobot import Groot
        from getiaction.policies.lerobot.groot import Groot as ExplicitGroot

        # Both should be the same class (explicit wrapper)
        assert Groot is ExplicitGroot

    @pytest.mark.skipif(
        not pytest.importorskip("lerobot", reason="LeRobot not installed"),
        reason="Requires lerobot",
    )
    def test_convenience_classes_available(self):
        """Test all convenience classes can be imported."""
        from getiaction.policies.lerobot import (
            ACT,
            Diffusion,
            Groot,
            PI0,
            PI05,
            PI0Fast,
            SAC,
            SmolVLA,
            TDMPC,
            VQBeT,
        )

        # All should be importable
        assert ACT is not None
        assert Diffusion is not None
        assert Groot is not None
        assert PI0 is not None
        assert PI05 is not None
        assert PI0Fast is not None
        assert SAC is not None
        assert SmolVLA is not None
        assert TDMPC is not None
        assert VQBeT is not None

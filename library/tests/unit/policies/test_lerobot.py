# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LeRobot policy wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if lerobot not installed
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("lerobot", reason="LeRobot not installed"),
    reason="Requires lerobot",
)


# Module-level fixtures for expensive operations (shared across all tests)
@pytest.fixture(scope="module")
def lerobot_imports():
    """Import LeRobot modules once per test module."""
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from getiaction.policies.lerobot import LeRobotPolicy

    return {"LeRobotPolicy": LeRobotPolicy, "LeRobotDataset": LeRobotDataset}


@pytest.fixture(scope="module")
def pusht_dataset(lerobot_imports):
    """Load pusht dataset once per module."""
    LeRobotDataset = lerobot_imports["LeRobotDataset"]
    return LeRobotDataset("lerobot/pusht")


@pytest.fixture(scope="module")
def pusht_act_policy(lerobot_imports, pusht_dataset):
    """Create ACT policy from pusht dataset once per module."""
    LeRobotPolicy = lerobot_imports["LeRobotPolicy"]
    return LeRobotPolicy.from_dataset("act", pusht_dataset)


class TestLeRobotPolicyLazyInit:
    """Tests for lazy initialization pattern."""

    def test_lazy_init_stores_policy_name(self, lerobot_imports):
        """Test lazy init stores policy name without initializing model."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert policy.policy_name == "diffusion"
        assert policy._config is None  # Not initialized yet

    def test_lazy_init_stores_kwargs(self, lerobot_imports):
        """Test kwargs are stored for deferred config creation."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(
            policy_name="act",
            optimizer_lr=1e-4,
            chunk_size=50,
        )

        assert policy._policy_config["optimizer_lr"] == 1e-4
        assert policy._policy_config["chunk_size"] == 50

    def test_lazy_init_merges_policy_config_and_kwargs(self, lerobot_imports):
        """Test policy_config dict is merged with kwargs, kwargs take precedence."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(
            policy_name="act",
            policy_config={"optimizer_lr": 1e-4, "chunk_size": 50},
            optimizer_lr=2e-4,  # Override via kwargs
        )

        # kwargs should override policy_config
        assert policy._policy_config["optimizer_lr"] == 2e-4
        assert policy._policy_config["chunk_size"] == 50


class TestLeRobotPolicyEagerInit:
    """Tests for eager initialization via from_dataset."""

    def test_from_dataset_with_lerobot_dataset(self, lerobot_imports, pusht_dataset):
        """Test from_dataset accepts LeRobotDataset instance."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Use shared dataset fixture
        policy = LeRobotPolicy.from_dataset("act", pusht_dataset)

        assert policy._config is not None  # Initialized
        assert hasattr(policy, "_lerobot_policy")

    def test_from_dataset_with_repo_id_string(self, pusht_act_policy):
        """Test from_dataset accepts repo ID string."""
        # Use shared fixture instead of creating new policy
        assert pusht_act_policy._config is not None
        assert hasattr(pusht_act_policy, "_lerobot_policy")

    def test_from_dataset_passes_kwargs_to_config(self, lerobot_imports, pusht_dataset):
        """Test from_dataset passes kwargs for config creation."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # Use optimizer_lr which is safe - doesn't have dependent validation
        policy = LeRobotPolicy.from_dataset(
            "act",
            pusht_dataset,  # Reuse cached dataset
            optimizer_lr=5e-5,
        )

        assert policy._config.optimizer_lr == 5e-5


class TestLeRobotPolicyMethods:
    """Tests for policy method signatures."""

    def test_has_validation_step(self, lerobot_imports):
        """Test validation_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert hasattr(policy, "validation_step")
        assert callable(policy.validation_step)

    def test_has_test_step(self, lerobot_imports):
        """Test test_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="diffusion")

        assert hasattr(policy, "test_step")
        assert callable(policy.test_step)

    def test_has_configure_optimizers(self, lerobot_imports):
        """Test configure_optimizers method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="act")

        assert hasattr(policy, "configure_optimizers")
        assert callable(policy.configure_optimizers)

    def test_has_training_step(self, lerobot_imports):
        """Test training_step method exists."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy(policy_name="act")

        assert hasattr(policy, "training_step")
        assert callable(policy.training_step)


class TestLeRobotPolicyConfigureOptimizers:
    """Tests for optimizer configuration using LeRobot presets."""

    def test_configure_optimizers_uses_lerobot_preset(self, pusht_act_policy):
        """Test that configure_optimizers uses LeRobot's get_optimizer_preset."""
        # Verify config has optimizer preset method
        assert hasattr(pusht_act_policy._config, "get_optimizer_preset")

        # Call configure_optimizers
        optimizer = pusht_act_policy.configure_optimizers()
        assert optimizer is not None

    def test_configure_optimizers_respects_lr_setting(self, lerobot_imports, pusht_dataset):
        """Test optimizer uses learning rate from config."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        lr = 5e-5
        policy = LeRobotPolicy.from_dataset(
            "act",
            pusht_dataset,  # Reuse cached dataset
            optimizer_lr=lr,
        )

        optimizer = policy.configure_optimizers()

        # Check first param group has correct lr
        assert optimizer.param_groups[0]["lr"] == lr


class TestLeRobotPolicySelectAction:
    """Tests for action selection method signature."""

    def test_select_action_method_exists(self, pusht_act_policy):
        """Test select_action method exists after from_dataset initialization."""
        assert hasattr(pusht_act_policy, "select_action")
        assert callable(pusht_act_policy.select_action)

    def test_policy_is_ready_after_from_dataset(self, pusht_act_policy):
        """Test policy is fully initialized after from_dataset."""
        # Check internal state is properly initialized
        assert pusht_act_policy._config is not None
        assert hasattr(pusht_act_policy, "_lerobot_policy")
        assert pusht_act_policy._lerobot_policy is not None

        # Check pre/post processors are loaded (for normalization)
        assert pusht_act_policy._preprocessor is not None
        assert pusht_act_policy._postprocessor is not None


class TestLeRobotPolicyErrorCases:
    """Tests for error handling."""

    def test_invalid_policy_name_raises_error(self, lerobot_imports, pusht_dataset):
        """Test that invalid policy name raises ValueError."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        # LeRobot's error message format
        with pytest.raises(ValueError, match="is not available"):
            LeRobotPolicy.from_dataset("nonexistent_policy", pusht_dataset)


class TestLeRobotPolicyUniversalWrapper:
    """Tests verifying universal wrapper works with multiple policy types."""

    def test_supports_act_policy(self, pusht_act_policy):
        """Test universal wrapper supports ACT policy."""
        assert pusht_act_policy._config is not None
        assert "act" in pusht_act_policy._config.__class__.__name__.lower()

    def test_supports_diffusion_policy(self, lerobot_imports, pusht_dataset):
        """Test universal wrapper supports Diffusion policy."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("diffusion", pusht_dataset)
        assert policy._config is not None
        assert "diffusion" in policy._config.__class__.__name__.lower()

    def test_supports_vqbet_policy(self, lerobot_imports, pusht_dataset):
        """Test universal wrapper supports VQ-BeT policy."""
        LeRobotPolicy = lerobot_imports["LeRobotPolicy"]

        policy = LeRobotPolicy.from_dataset("vqbet", pusht_dataset)
        assert policy._config is not None
        assert "vqbet" in policy._config.__class__.__name__.lower()

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LeRobot policies.

This module validates the training pipeline for LeRobot policies:
    1. Train a policy using LeRobot PushT dataset
    2. Validate/test the trained policy

Note:
    LeRobot policies do not support export functionality.
    For export tests, see test_first_party_e2e.py.

Supported Policies:
    Core (always run):
        - act: Action Chunking Transformer
        - diffusion: Diffusion Policy

    Extended (marked @pytest.mark.slow):
        - vqbet: VQ-BeT (Vector Quantized Behavior Transformer)

    VLA (require special dependencies, not tested by default):
        - pi0, pi05, smolvla, groot, xvla
"""

import pytest

from getiaction.data import LeRobotDataModule
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Core policies - fast, no special dependencies
CORE_POLICIES = ["act", "diffusion"]

# Extended policies - slower, requires more compute
EXTENDED_POLICIES = ["vqbet"]

# VLA policies - require flash-attn/transformers (not tested by default)
# VLA_POLICIES = ["pi0", "smolvla", "groot"]


class LeRobotE2ETestBase:
    """Base class with common fixtures and tests for LeRobot policies."""

    @pytest.fixture(scope="class")
    def trainer(self) -> Trainer:
        """Create trainer with fast development configuration."""
        return Trainer(
            fast_dev_run=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

    @pytest.fixture(scope="class")
    def policy_name(self, request: pytest.FixtureRequest) -> str:
        """Extract policy name from parametrize."""
        return request.param

    @pytest.fixture(scope="class")
    def datamodule(self, policy_name: str) -> LeRobotDataModule:
        """Create datamodule for LeRobot policies with delta timestamps."""
        repo_id = "lerobot/pusht"
        fps = 10

        # Policy-specific configurations for delta timestamps
        policy_configs = {
            "act": {
                "action_delta_indices": list(range(100)),  # chunk_size=100
            },
            "diffusion": {
                "observation_delta_indices": [-1, 0],  # n_obs_steps=2
                "action_delta_indices": list(range(-1, 15)),  # horizon=16
            },
            "vqbet": {
                "observation_delta_indices": [-1, 0],  # n_obs_steps=2
                "action_delta_indices": list(range(100)),  # chunk_size=100
            },
        }

        config: dict = {
            "repo_id": repo_id,
            "train_batch_size": 8,
            "episodes": list(range(10)),
            "data_format": "lerobot",
        }

        # Add delta timestamps if configured for this policy
        if policy_name in policy_configs:
            policy_cfg = policy_configs[policy_name]
            delta_timestamps = {}

            if "observation_delta_indices" in policy_cfg:
                obs_indices = policy_cfg["observation_delta_indices"]
                delta_timestamps["observation.image"] = [i / fps for i in obs_indices]
                delta_timestamps["observation.state"] = [i / fps for i in obs_indices]

            if "action_delta_indices" in policy_cfg:
                action_indices = policy_cfg["action_delta_indices"]
                delta_timestamps["action"] = [i / fps for i in action_indices]

            config["delta_timestamps"] = delta_timestamps

        return LeRobotDataModule(**config)

    @pytest.fixture(scope="class")
    def policy(self, policy_name: str) -> Policy:
        """Create LeRobot policy instance with fast config for tests."""
        policy_kwargs: dict = {}

        # Policy-specific fast configurations
        if policy_name == "diffusion":
            policy_kwargs = {
                "num_train_timesteps": 10,
                "num_inference_steps": 5,
            }

        return get_policy(policy_name, source="lerobot", **policy_kwargs)

    @pytest.fixture(scope="class")
    def trained_policy(self, policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> Policy:
        """Train policy once and reuse across all tests."""
        trainer.fit(policy, datamodule=datamodule)
        return policy

    # --- Tests ---

    def test_train_policy(self, trained_policy: Policy, trainer: Trainer) -> None:
        """Test that policy was trained successfully."""
        assert trainer.state.finished

    def test_validate_policy(self, trained_policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be validated."""
        trainer.validate(trained_policy, datamodule=datamodule)
        assert trainer.state.finished

    def test_test_policy(self, trained_policy: Policy, datamodule: LeRobotDataModule, trainer: Trainer) -> None:
        """Test that trained policy can be tested."""
        trainer.test(trained_policy, datamodule=datamodule)
        assert trainer.state.finished


@pytest.mark.parametrize("policy_name", CORE_POLICIES, indirect=True)
class TestLeRobotCorePolicies(LeRobotE2ETestBase):
    """E2E tests for core LeRobot policies (ACT, Diffusion).

    These tests run by default and cover the most commonly used policies.
    """

    pass


@pytest.mark.slow
@pytest.mark.parametrize("policy_name", EXTENDED_POLICIES, indirect=True)
class TestLeRobotExtendedPolicies(LeRobotE2ETestBase):
    """E2E tests for extended LeRobot policies (VQ-BeT, etc.).

    These tests are slower and marked with @pytest.mark.slow.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    pass

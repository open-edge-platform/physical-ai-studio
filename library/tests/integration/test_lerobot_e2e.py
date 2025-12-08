# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LeRobot policies.

This module validates the training pipeline for LeRobot policies:
    1. Train a policy using LeRobot ALOHA dataset
    2. Validate/test the trained policy

Note:
    LeRobot policies do not support export functionality.
    For export tests, see test_first_party_e2e.py.

Supported Policies:
    Core (always run):
        - act: Action Chunking Transformer
        - diffusion: Diffusion Policy

    Extended (marked @pytest.mark.slow):
        - vqbet: Vector Quantized Behavior Transformer

    VLA (marked @pytest.mark.slow, requires 24GB+ VRAM):
        - groot: NVIDIA GR00T-N1.5-3B (trains projector + action head only)

    Not tested (no explicit wrappers yet):
        - pi0, pi05, smolvla, xvla, tdmpc, sac, reward_classifier
"""

import pytest

from getiaction.data import LeRobotDataModule
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Core policies - fast, no special dependencies
CORE_POLICIES = ["act", "diffusion"]

# Extended policies - slower but still practical
EXTENDED_POLICIES = ["vqbet"]

# VLA policies - large models requiring 24GB+ VRAM
VLA_POLICIES = ["groot"]


def get_delta_timestamps_from_policy(policy_name: str, fps: int = 10) -> dict[str, list[float]]:
    """Derive delta timestamps configuration from LeRobot policy config.

    This extracts n_obs_steps and action chunk/horizon size from the policy's
    default configuration to automatically compute the correct delta timestamps.

    Args:
        policy_name: Name of the LeRobot policy (e.g., "act", "diffusion").
        fps: Frames per second of the dataset.

    Returns:
        Dictionary with delta timestamps for observation and action keys.
    """
    from lerobot.policies.factory import make_policy_config

    config = make_policy_config(policy_name)

    n_obs_steps = getattr(config, "n_obs_steps", 1)

    # Get action sequence length - different policies use different attribute names
    action_length = (
        getattr(config, "chunk_size", None)
        or getattr(config, "horizon", None)
        or getattr(config, "action_chunk_size", None)
        or getattr(config, "n_action_steps", 1)
    )

    delta_timestamps: dict[str, list[float]] = {}

    # Observation timestamps: indices from -(n_obs_steps-1) to 0
    if n_obs_steps > 1:
        obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
        delta_timestamps["observation.images.top"] = [i / fps for i in obs_indices]
        delta_timestamps["observation.state"] = [i / fps for i in obs_indices]

    # Action timestamps: depends on policy type
    if policy_name == "diffusion":
        # Diffusion predicts horizon steps starting from -1
        action_indices = list(range(-1, action_length - 1))
    else:
        # Other policies predict chunk_size steps starting from 0
        action_indices = list(range(action_length))

    delta_timestamps["action"] = [i / fps for i in action_indices]

    return delta_timestamps


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
        """Create datamodule for LeRobot policies with delta timestamps derived from policy config."""
        delta_timestamps = get_delta_timestamps_from_policy(policy_name)

        return LeRobotDataModule(
            repo_id="lerobot/aloha_sim_insertion_human",
            train_batch_size=8,
            episodes=list(range(10)),
            data_format="lerobot",
            delta_timestamps=delta_timestamps if delta_timestamps else None,
        )

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
    """E2E tests for extended LeRobot policies (VQ-BeT).

    These tests are slower and marked with @pytest.mark.slow.
    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    pass


@pytest.mark.slow
@pytest.mark.parametrize("policy_name", VLA_POLICIES, indirect=True)
class TestLeRobotVLAPolicies(LeRobotE2ETestBase):
    """E2E tests for Vision-Language-Action policies (groot).

    These tests require:
    - 24GB+ VRAM (48GB recommended)
    - flash_attn package (CUDA only): pip install flash-attn
    - peft package: pip install peft

    By default, Groot freezes the backbone and only trains the projector + action head.

    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class", autouse=True)
    def check_flash_attn(self) -> None:
        """Skip if flash_attn is not available."""
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pytest.skip("Groot requires flash_attn: pip install flash-attn")

    pass

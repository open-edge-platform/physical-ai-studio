# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LeRobot policies with explicit wrappers.

This module validates the training pipeline for LeRobot policies that have
explicit wrappers (ACT, Diffusion, Groot) - not the universal LeRobotPolicy.

Workflow:
    1. Train a policy using LeRobot ALOHA dataset
    2. Validate/test the trained policy

Note:
    LeRobot policies do not support export functionality.
    For export tests, see test_first_party_e2e.py.

Tested Policies (have explicit wrappers):
    Core (always run):
        - act: Action Chunking Transformer
        - diffusion: Diffusion Policy

    VLA (marked @pytest.mark.slow, requires flash_attn + 24GB+ VRAM):
        - groot: NVIDIA GR00T-N1.5-3B (trains projector + action head only)
"""

import pytest

from getiaction.data import LeRobotDataModule
from getiaction.data.lerobot import get_delta_timestamps_from_policy
from getiaction.policies import get_policy
from getiaction.policies.base.policy import Policy
from getiaction.train import Trainer

# Core policies - fast, have explicit wrappers
CORE_POLICIES = ["act", "diffusion"]

# VLA policies - large models requiring flash_attn + 24GB+ VRAM
VLA_POLICIES = ["groot"]


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
@pytest.mark.skip(
    reason="LeRobot Groot wrapper has action dimension validation issues. "
    "Use first-party getiaction Groot implementation instead (test_first_party_e2e.py)."
)
@pytest.mark.parametrize("policy_name", VLA_POLICIES, indirect=True)
class TestLeRobotVLAPolicies(LeRobotE2ETestBase):
    """E2E tests for Vision-Language-Action policies (groot).

    NOTE: These tests are skipped because LeRobot's Groot wrapper has known issues
    with action dimension validation. Use the first-party getiaction Groot
    implementation instead, which is tested in test_first_party_e2e.py.

    These tests require:
    - 24GB+ VRAM

    By default, Groot freezes the backbone and only trains the projector + action head.

    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class", autouse=True)
    def check_groot_dependencies(self) -> None:
        """Skip if lerobot[groot] dependencies are not available.

        Groot requires: pip install 'lerobot[groot]'
        This includes flash-attn (CUDA-only), peft, transformers, etc.
        """
        from lerobot.utils.import_utils import is_package_available

        if not is_package_available("flash_attn"):
            pytest.skip("Groot requires lerobot[groot]: uv pip install 'lerobot[groot]'")

    @pytest.fixture(scope="class")
    def datamodule(self, policy_name: str) -> LeRobotDataModule:
        """Create datamodule for VLA policies with smaller batch size for memory."""
        delta_timestamps = get_delta_timestamps_from_policy(policy_name)

        return LeRobotDataModule(
            repo_id="lerobot/aloha_sim_insertion_human",
            train_batch_size=1,  # Small batch for 24GB GPU memory
            episodes=list(range(2)),
            data_format="lerobot",
            delta_timestamps=delta_timestamps if delta_timestamps else None,
        )

    @pytest.fixture(scope="class")
    def policy(self, policy_name: str) -> Policy:
        """Create VLA policy with memory-efficient settings for 24GB GPUs."""
        if policy_name == "groot":
            # Memory-efficient settings: freeze backbone, only train projector
            return get_policy(
                policy_name,
                source="lerobot",
                tune_llm=False,
                tune_visual=False,
                tune_projector=True,
                tune_diffusion_model=False,
            )
        return get_policy(policy_name, source="lerobot")

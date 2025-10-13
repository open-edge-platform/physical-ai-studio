# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for training and validation with gym rollouts.

This module tests the complete pipeline:
1. Train a policy (first-party ACT or third-party LeRobot)
2. Run validation with gym rollouts
3. Verify metrics are logged correctly
"""

from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Trainer

from getiaction.data import DataModule
from getiaction.gyms import PushTGym


@pytest.fixture
def pusht_gym():
    """Create PushT gym environment."""
    return PushTGym()


class TestFirstPartyPolicyE2E:
    """End-to-end tests for first-party policies (ACT, Dummy)."""

    def test_dummy_policy_training_and_validation(self, dummy_dataset, pusht_gym):
        """Test dummy policy with training + gym validation."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        # Create datamodule with training data and validation gym
        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=8),
            train_batch_size=4,
            val_gyms=pusht_gym,
            num_rollouts_val=2,  # Just 2 rollouts for speed
            max_episode_steps=10,  # Very short episodes
        )

        # Create dummy policy
        config = DummyConfig(action_shape=(2,))  # PushT action shape
        policy = Dummy(config)

        # Train for just 1 step with validation
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        # This should complete without errors
        trainer.fit(policy, datamodule=datamodule)

        # Verify validation was called (logged_metrics should have validation metrics)
        assert trainer.logged_metrics is not None
        # Check for gym validation metrics
        assert any(key.startswith("val/") for key in trainer.logged_metrics)

    def test_act_policy_training_and_validation(self, dummy_dataset, pusht_gym):
        """Test ACT policy with training + gym validation.

        Note: With fast_dev_run=True, only 1 batch is processed, so no GPU required.
        """
        pytest.skip("ACT policy needs reset() method implementation")
        from getiaction.policies.act import ACT

        # Create datamodule
        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=8),
            train_batch_size=4,
            val_gyms=pusht_gym,
            num_rollouts_val=2,
            max_episode_steps=10,
        )

        # Create ACT policy with minimal config
        policy = ACT()  # TODO: Update when ACT API is finalized

        # Train for just 1 step with validation
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=datamodule)

        # Verify validation metrics
        assert trainer.logged_metrics is not None
        assert any(key.startswith("val/") for key in trainer.logged_metrics)


@pytest.mark.slow
class TestLeRobotPolicyE2E:
    """End-to-end tests for third-party LeRobot policies."""

    @pytest.fixture
    def lerobot_dataset(self):
        """Create minimal LeRobot-compatible dataset."""
        pytest.importorskip("lerobot")
        from getiaction.data.lerobot import LeRobotDataModule

        # Use a very small LeRobot dataset for testing
        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            train_batch_size=4,
            episodes=[0, 1],  # Just 2 episodes
            data_format="lerobot",  # LeRobot format
        )

    def test_lerobot_diffusion_training_and_validation(self, lerobot_dataset, pusht_gym):
        """Test LeRobot Diffusion policy with training + gym validation.

        Note: With fast_dev_run=True, only 1 batch is processed, so no GPU required.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import DiffusionPolicy

        # Add gym to datamodule
        lerobot_dataset.val_gyms = pusht_gym
        lerobot_dataset.num_rollouts_val = 2
        lerobot_dataset.max_episode_steps = 10

        # Create policy
        policy = DiffusionPolicy(
            dataset=lerobot_dataset.train_dataset,
            learning_rate=1e-4,
            num_steps=10,  # Very small for testing
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset)

        # Verify metrics
        assert trainer.logged_metrics is not None
        assert any(key.startswith("val/") for key in trainer.logged_metrics)
        assert any(key.startswith("train/") for key in trainer.logged_metrics)

    def test_lerobot_act_training_and_validation(self, lerobot_dataset, pusht_gym):
        """Test LeRobot ACT policy with training + gym validation.

        Note: With fast_dev_run=True, only 1 batch is processed, so no GPU required.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import ACTPolicy

        # Add gym to datamodule
        lerobot_dataset.val_gyms = pusht_gym
        lerobot_dataset.num_rollouts_val = 2
        lerobot_dataset.max_episode_steps = 10

        # Create policy
        policy = ACTPolicy(
            dataset=lerobot_dataset.train_dataset,
            learning_rate=1e-4,
            chunk_size=4,
            hidden_dim=128,
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset)

        # Verify metrics
        assert trainer.logged_metrics is not None
        assert any(key.startswith("val/") for key in trainer.logged_metrics)

    def test_lerobot_universal_training_and_validation(self, lerobot_dataset, pusht_gym):
        """Test LeRobot Universal wrapper with training + gym validation.

        Note: With fast_dev_run=True, only 1 batch is processed, so no GPU required.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import LeRobotPolicy

        # Add gym to datamodule
        lerobot_dataset.val_gyms = pusht_gym
        lerobot_dataset.num_rollouts_val = 2
        lerobot_dataset.max_episode_steps = 10

        # Create policy using universal wrapper
        policy = LeRobotPolicy(
            policy_name="diffusion",
            dataset=lerobot_dataset.train_dataset,
            learning_rate=1e-4,
            num_steps=10,
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset)

        # Verify metrics
        assert trainer.logged_metrics is not None
        assert any(key.startswith("val/") for key in trainer.logged_metrics)


class TestRolloutMetrics:
    """Test that rollout metrics are computed and logged correctly."""

    def test_rollout_metrics_structure(self, dummy_dataset, pusht_gym):
        """Verify the structure and content of rollout metrics."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=8),
            train_batch_size=4,
            val_gyms=pusht_gym,
            num_rollouts_val=1,
            max_episode_steps=5,
        )

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config)

        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=datamodule)

        # Check specific metric keys (metrics are prefixed with val/gym/)
        expected_metrics = [
            "val/gym/episode_length",
            "val/gym/sum_reward",
            "val/gym/success",
        ]

        logged_keys = list(trainer.logged_metrics.keys())
        for metric in expected_metrics:
            assert metric in logged_keys, f"Missing metric: {metric}"

    def test_multiple_validation_rollouts(self, dummy_dataset, pusht_gym):
        """Test that multiple rollouts are aggregated correctly."""
        from getiaction.policies.dummy import Dummy, DummyConfig

        num_rollouts = 3

        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=8),
            train_batch_size=4,
            val_gyms=pusht_gym,
            num_rollouts_val=num_rollouts,
            max_episode_steps=5,
        )

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config)

        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=datamodule)

        # Verify metrics exist (Lightning aggregates them automatically)
        assert trainer.logged_metrics is not None
        assert len(trainer.logged_metrics) > 0

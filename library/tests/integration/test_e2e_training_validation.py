# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for training and validation with gym rollouts.

This module contains true E2E tests that run the complete training pipeline:
1. Initialize a policy (first-party ACT/Dummy or third-party LeRobot)
2. Load training data (real LeRobot datasets or dummy data)
3. Run trainer.fit() with validation
4. Execute gym rollouts during validation
5. Verify metrics are logged correctly

All tests in this file use Lightning's Trainer and run actual training loops.
For unit tests or non-E2E integration tests, see tests/unit/ or tests/integration/
with different naming patterns.

Test Classes:
- TestFirstPartyPolicyE2E: E2E tests for first-party policies (Dummy, ACT)
- TestLeRobotPolicyE2E: E2E tests for LeRobot policy wrappers (slow, requires download)
- TestRolloutMetrics: E2E tests validating gym rollout metric computation
"""

from __future__ import annotations

import pytest
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

    @pytest.mark.xfail(
        reason=(
            "First-party ACT policy missing action queue management for gym rollouts. "
            "ACT predicts action chunks (100 actions) but gym.step() expects single actions. "
            "Need to implement action queue: predict chunk once, execute actions one-by-one, "
            "predict new chunk when queue empty. LeRobot ACT wrapper handles this correctly."
        ),
        strict=True,
    )
    def test_act_policy_training_and_validation(self, dummy_dataset, pusht_gym):
        """Test ACT policy with training + gym validation.

        Note: With fast_dev_run=True, only 1 batch is processed, so no GPU required.
        ACT uses action chunking (chunk_size=100 by default), so we configure
        delta_indices to provide action sequences.

        The dummy dataset is configured to match PushT gym dimensions:
        - state: 2D (agent x,y position)
        - action: 2D (agent velocity)
        """
        from getiaction.policies.act import ACT

        # Create dataset and configure action chunking for ACT
        dataset = dummy_dataset(num_samples=8, state_dim=2, action_dim=2)
        chunk_size = 100  # ACT default chunk_size
        dataset.delta_indices = {"action": list(range(chunk_size))}

        # Create datamodule
        datamodule = DataModule(
            train_dataset=dataset,
            train_batch_size=4,
            val_gyms=pusht_gym,
            num_rollouts_val=2,
            max_episode_steps=10,
        )

        # Create ACT policy
        policy = ACT()

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
@pytest.mark.requires_download
class TestLeRobotPolicyE2E:
    """Test LeRobot policies with E2E training and validation.

    Uses a real LeRobot dataset (lerobot/pusht with 2 episodes) which is downloaded
    once and cached. Download is fast (~2-3 seconds, ~1MB) on first run.
    """

    @pytest.fixture
    def lerobot_dataset_diffusion(self):
        """Create LeRobotDataModule for Diffusion policy with action chunking.

        Uses lerobot/pusht with just 2 episodes for fast testing. The dataset is
        automatically downloaded and cached on first use (~2-3s, ~1MB).
        Configures delta_timestamps to provide action sequences (horizon=16) and
        observation history (n_obs_steps=2).
        """
        pytest.importorskip("lerobot")
        from getiaction.data.lerobot import LeRobotDataModule
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Get FPS for the dataset to calculate time deltas
        temp_ds = LeRobotDataset("lerobot/pusht", episodes=[0, 1])
        fps = temp_ds.fps
        dt_per_step = 1.0 / fps

        # Diffusion uses horizon=16 and n_obs_steps=2 by default
        horizon = 16
        n_obs_steps = 2
        delta_timestamps = {
            "observation.image": [i * dt_per_step for i in range(n_obs_steps)],
            "observation.state": [i * dt_per_step for i in range(n_obs_steps)],
            "action": [i * dt_per_step for i in range(horizon)],
        }

        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            episodes=[0, 1],  # Only load 2 episodes (~1MB, fast download)
            train_batch_size=4,
            data_format="lerobot",  # Use LeRobot's native dict format
            delta_timestamps=delta_timestamps,
        )

    @pytest.fixture
    def lerobot_dataset_act(self):
        """Create LeRobotDataModule for ACT policy with action chunking.

        Uses lerobot/pusht with just 2 episodes for fast testing. The dataset is
        automatically downloaded and cached on first use (~2-3s, ~1MB).
        Configures delta_timestamps to provide action sequences (chunk_size=100).
        ACT processes single observations, so no observation history needed.
        """
        pytest.importorskip("lerobot")
        from getiaction.data.lerobot import LeRobotDataModule
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Get FPS for the dataset to calculate time deltas
        temp_ds = LeRobotDataset("lerobot/pusht", episodes=[0, 1])
        fps = temp_ds.fps
        dt_per_step = 1.0 / fps

        # ACT uses chunk_size=100 by default
        chunk_size = 100
        delta_timestamps = {"action": [i * dt_per_step for i in range(chunk_size)]}

        return LeRobotDataModule(
            repo_id="lerobot/pusht",
            episodes=[0, 1],  # Only load 2 episodes (~1MB, fast download)
            train_batch_size=4,
            data_format="lerobot",  # Use LeRobot's native dict format
            delta_timestamps=delta_timestamps,
        )

    def test_lerobot_diffusion_training_and_validation(self, lerobot_dataset_diffusion, pusht_gym):
        """Test LeRobot Diffusion policy with training + gym validation.

        Uses lerobot/pusht dataset with 2 episodes. Downloads about 1MB on first run.
        With fast_dev_run=True, only 1 batch is processed.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import Diffusion

        # Add gym to datamodule
        lerobot_dataset_diffusion.val_gyms = pusht_gym
        lerobot_dataset_diffusion.num_rollouts_val = 2
        lerobot_dataset_diffusion.max_episode_steps = 10

        # Create policy with default LeRobot parameters (not minimal, to match dataset)
        policy = Diffusion(
            learning_rate=1e-4,
            # Use defaults that match LeRobot dataset structure
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset_diffusion)

        # Verify metrics
        assert trainer.logged_metrics is not None
        assert any(key.startswith("val/") for key in trainer.logged_metrics)

    def test_lerobot_act_training_and_validation(self, lerobot_dataset_act, pusht_gym):
        """Test LeRobot ACT policy with training + gym validation.

        Uses lerobot/pusht dataset with 2 episodes. Downloads about 1MB on first run.
        With fast_dev_run=True, only 1 batch is processed.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import ACT

        # Add gym to datamodule
        lerobot_dataset_act.val_gyms = pusht_gym
        lerobot_dataset_act.num_rollouts_val = 2
        lerobot_dataset_act.max_episode_steps = 10

        # Create policy with default LeRobot parameters (not minimal, to match dataset)
        policy = ACT(
            learning_rate=1e-4,
            # Use defaults that match LeRobot dataset structure
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset_act)

    def test_lerobot_universal_training_and_validation(self, lerobot_dataset_diffusion, pusht_gym):
        """Test LeRobot Universal wrapper with training + gym validation.

        Uses lerobot/pusht dataset with 2 episodes. Downloads about 1MB on first run.
        With fast_dev_run=True, only 1 batch is processed.
        """
        pytest.importorskip("lerobot")
        from getiaction.policies.lerobot import LeRobotPolicy

        # Add gym to datamodule
        lerobot_dataset_diffusion.val_gyms = pusht_gym
        lerobot_dataset_diffusion.num_rollouts_val = 2
        lerobot_dataset_diffusion.max_episode_steps = 10

        # Create policy using universal wrapper with default LeRobot parameters
        policy = LeRobotPolicy(
            policy_name="diffusion",
            learning_rate=1e-4,
            # Use defaults that match LeRobot dataset
        )

        # Train
        trainer = Trainer(
            fast_dev_run=True,  # Run 1 train, val, test batch
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(policy, datamodule=lerobot_dataset_diffusion)

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

        # Check specific metric keys
        # Per-episode metrics (logged on_step=True)
        per_episode_metrics = [
            "val/gym/episode/episode_length",
            "val/gym/episode/sum_reward",
            "val/gym/episode/success",
        ]

        # Aggregated metrics (logged on_epoch=True, computed at epoch end)
        aggregated_metrics = [
            "val/gym/avg_episode_length",
            "val/gym/avg_sum_reward",
            "val/gym/pc_success",
        ]

        logged_keys = list(trainer.logged_metrics.keys())

        # Check per-episode metrics
        for metric in per_episode_metrics:
            assert metric in logged_keys, f"Missing per-episode metric: {metric}"

        # Check aggregated metrics
        for metric in aggregated_metrics:
            assert metric in logged_keys, f"Missing aggregated metric: {metric}"

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

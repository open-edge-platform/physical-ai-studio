# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for Rollout to ensure numerical equivalence with evaluate_policy."""

from __future__ import annotations

import pytest
import torch

from getiaction.eval import Rollout, evaluate_policy


@pytest.fixture
def dummy_policy():
    """Create a dummy policy for testing."""
    from getiaction.policies.dummy import Dummy, DummyConfig

    return Dummy(config=DummyConfig(action_shape=(2,)))


class TestRolloutNumericalEquivalence:
    """Tests that verify Rollout produces identical results to evaluate_policy."""

    @pytest.mark.parametrize(("n_episodes", "seed"), [(1, 42), (5, 100)])
    def test_aggregated_metrics_equivalence(
        self, pusht_gym, dummy_policy, n_episodes, seed
    ):
        """Verify Rollout produces same aggregated metrics as evaluate_policy."""
        max_steps = 30

        # Old implementation
        old_results = evaluate_policy(
            env=pusht_gym,
            policy=dummy_policy,
            n_episodes=n_episodes,
            start_seed=seed,
            max_steps=max_steps,
        )

        # New implementation
        metric = Rollout(max_steps=max_steps)
        for i in range(n_episodes):
            metric.update(env=pusht_gym, policy=dummy_policy, seed=seed + i)
        new_results = metric.compute()

        # Verify all metrics match (0.1% tolerance for floating point differences)
        for key in ["avg_sum_reward", "avg_max_reward", "avg_episode_length"]:
            assert torch.isclose(
                new_results[key],
                torch.tensor(old_results["aggregated"][key]),
                rtol=1e-3,
                atol=1e-5,
            ), f"{key} mismatch"

    def test_per_episode_data_matches(self, pusht_gym, dummy_policy):
        """Verify per-episode data matches between implementations."""
        n_episodes, seed, max_steps = 3, 200, 20

        # Old implementation
        old_results = evaluate_policy(
            env=pusht_gym,
            policy=dummy_policy,
            n_episodes=n_episodes,
            start_seed=seed,
            max_steps=max_steps,
        )

        # New implementation
        metric = Rollout(max_steps=max_steps)
        for i in range(n_episodes):
            metric.update(env=pusht_gym, policy=dummy_policy, seed=seed + i)
        per_episode_new = metric.get_per_episode_data()

        # Verify per-episode data
        assert len(per_episode_new) == len(old_results["per_episode"]) == n_episodes
        for i, (old_ep, new_ep) in enumerate(
            zip(old_results["per_episode"], per_episode_new)
        ):
            assert abs(new_ep["sum_reward"] - old_ep["sum_reward"]) < 1e-5
            assert abs(new_ep["max_reward"] - old_ep["max_reward"]) < 1e-5
            assert new_ep["episode_length"] == old_ep["episode_length"]


class TestRolloutBehavior:
    """Tests for Rollout metric behavior and state management."""

    def test_metric_reset(self, pusht_gym, dummy_policy):
        """Verify metric reset clears state correctly."""
        metric = Rollout(max_steps=20)

        # Accumulate data
        for i in range(3):
            metric.update(env=pusht_gym, policy=dummy_policy, seed=i)

        assert metric.num_episodes == 3 and metric.sum_rewards > 0

        # Reset and verify empty state
        metric.reset()
        assert metric.num_episodes == 0 and metric.sum_rewards == 0
        assert len(metric.all_sum_rewards) == 0  # type: ignore[arg-type]

        results = metric.compute()
        assert results["avg_sum_reward"] == 0.0 and results["n_episodes"] == 0

    def test_empty_metric_compute(self):
        """Verify compute() handles empty state correctly."""
        results = Rollout().compute()
        assert all(
            results[k] == 0.0
            for k in ["avg_sum_reward", "avg_max_reward", "avg_episode_length"]
        )
        assert results["n_episodes"] == 0

    def test_metric_state_types(self, pusht_gym, dummy_policy):
        """Verify metric state and output types."""
        metric = Rollout(max_steps=5)
        metric.update(env=pusht_gym, policy=dummy_policy, seed=999)

        # State is correct
        assert metric.num_episodes == 1
        assert len(metric.all_sum_rewards) == 1  # type: ignore[arg-type]
        assert isinstance(metric.all_sum_rewards[0], torch.Tensor)  # type: ignore[index]

        # Output is correct
        results = metric.compute()
        assert isinstance(results, dict)
        assert all(isinstance(v, torch.Tensor) for v in results.values())

    def test_multiple_compute_calls_idempotent(self, pusht_gym, dummy_policy):
        """Verify compute() is idempotent (no side effects)."""
        metric = Rollout(max_steps=20)
        for i in range(3):
            metric.update(env=pusht_gym, policy=dummy_policy, seed=i)

        # Multiple compute calls should return identical results
        results = [metric.compute() for _ in range(3)]
        for key in ["avg_sum_reward", "avg_episode_length"]:
            assert all(torch.equal(results[0][key], r[key]) for r in results[1:])


class TestRolloutIntegration:
    """Integration tests for Rollout with Lightning."""

    def test_lightning_training_integration(self, dummy_dataset, pusht_gym):
        """Verify Rollout works correctly in Lightning training loop."""
        from lightning.pytorch import Trainer

        from getiaction.data import DataModule
        from getiaction.policies.dummy import Dummy, DummyConfig

        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=8),
            train_batch_size=4,
            val_gym=pusht_gym,
            num_rollouts_val=3,
            max_episode_steps=10,
        )

        policy = Dummy(config=DummyConfig(action_shape=(2,)))
        trainer = Trainer(
            fast_dev_run=True,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(policy, datamodule=datamodule)

        # Verify aggregated metrics were logged
        metric_keys = list(trainer.logged_metrics.keys())
        assert any("val/gym/avg_sum_reward" in k for k in metric_keys)

    def test_device_placement(self, pusht_gym, dummy_policy):
        """Verify metric tensors are on correct device."""
        metric = Rollout(max_steps=10)
        metric.update(env=pusht_gym, policy=dummy_policy, seed=0)

        # All tensors on CPU by default
        assert metric.sum_rewards.device.type == "cpu"  # type: ignore[union-attr]
        assert metric.all_sum_rewards[0].device.type == "cpu"  # type: ignore[index,union-attr]
        assert metric.compute()["avg_sum_reward"].device.type == "cpu"

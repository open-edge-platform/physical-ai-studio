# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rollout functionality."""

from __future__ import annotations

import pytest


class TestRollout:
    """Tests for the rollout function."""

    def test_rollout_executes_successfully(self):
        """Test that rollout function executes and returns correct structure."""
        from getiaction.eval import rollout
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        result = rollout(
            env=gym,
            policy=policy,
            seed=42,
            max_steps=5,  # Very short for speed
            return_observations=False,
        )

        # Verify result structure
        assert "episode_length" in result
        assert "sum_reward" in result
        assert "max_reward" in result
        assert "is_success" in result

    def test_rollout_return_types(self):
        """Test that rollout returns correct types."""
        from getiaction.eval import rollout
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym = PushTGym()

        result = rollout(
            env=gym,
            policy=policy,
            seed=42,
            max_steps=5,
            return_observations=False,
        )

        # Verify types
        assert isinstance(result["episode_length"], int)
        assert isinstance(result["sum_reward"], float)
        assert isinstance(result["is_success"], bool)

    def test_rollout_with_seed_reproducibility(self):
        """Test that rollout is reproducible with same seed."""
        from getiaction.eval import rollout
        from getiaction.gyms import PushTGym
        from getiaction.policies.dummy import Dummy, DummyConfig

        config = DummyConfig(action_shape=(2,))
        policy = Dummy(config=config)
        gym1 = PushTGym()
        gym2 = PushTGym()

        result1 = rollout(env=gym1, policy=policy, seed=42, max_steps=5, return_observations=False)
        result2 = rollout(env=gym2, policy=policy, seed=42, max_steps=5, return_observations=False)

        # Same seed should give same results
        assert result1["episode_length"] == result2["episode_length"]

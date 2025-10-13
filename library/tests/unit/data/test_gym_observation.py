# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GymObservation dataclass."""

from unittest.mock import Mock

import pytest
import torch

from getiaction.data.observation import GymObservation


class TestGymObservationCreation:
    """Test GymObservation instantiation and basic properties."""

    def test_basic_creation(self) -> None:
        """Test basic GymObservation creation with all parameters."""
        env = Mock()
        env.name = "test-env"

        obs = GymObservation(env=env, episode_id=42, seed=123, max_steps=100)

        assert obs.env is env
        assert obs.episode_id == 42
        assert obs.seed == 123
        assert obs.max_steps == 100

    def test_creation_with_defaults(self) -> None:
        """Test GymObservation with default values."""
        env = Mock()
        env.name = "test-env"

        obs = GymObservation(env=env)

        assert obs.env is env
        assert obs.episode_id == 0
        assert obs.seed is None
        assert obs.max_steps is None


class TestGymObservationRepresentation:
    """Test GymObservation string representation."""

    def test_repr_with_all_fields(self) -> None:
        """Test GymObservation string representation with all fields."""
        env = Mock()
        env._gym_id = "test-env"

        obs = GymObservation(env=env, episode_id=5, seed=999, max_steps=50)
        repr_str = repr(obs)

        assert "GymObservation" in repr_str
        assert "env=test-env" in repr_str
        assert "episode_id=5" in repr_str
        assert "seed=999" in repr_str
        assert "max_steps=50" in repr_str

    def test_repr_with_none_values(self) -> None:
        """Test repr with None values doesn't show them."""
        env = Mock()
        env._gym_id = "test-env"

        obs = GymObservation(env=env)
        repr_str = repr(obs)

        assert "GymObservation" in repr_str
        assert "env=test-env" in repr_str
        assert "episode_id=0" in repr_str
        # When None, seed and max_steps shouldn't appear in repr
        assert "seed=None" not in repr_str
        assert "max_steps=None" not in repr_str

    def test_repr_with_unknown_env(self) -> None:
        """Test repr when environment doesn't have _gym_id."""
        env = Mock(spec=[])  # Mock without _gym_id attribute

        obs = GymObservation(env=env, episode_id=1)
        repr_str = repr(obs)

        assert "GymObservation" in repr_str
        assert "env=unknown" in repr_str
        assert "episode_id=1" in repr_str


class TestGymObservationTypeChecking:
    """Test GymObservation type checking and isinstance behavior."""

    def test_isinstance_check(self) -> None:
        """Test that isinstance checks work correctly."""
        env = Mock()
        obs = GymObservation(env=env)

        assert isinstance(obs, GymObservation)

    def test_not_a_dict(self) -> None:
        """Test that GymObservation is not a dict."""
        env = Mock()
        obs = GymObservation(env=env)

        # Should not be a dict (important for batch type checking)
        assert not isinstance(obs, dict)

    def test_not_a_tensor(self) -> None:
        """Test that GymObservation is not a tensor."""
        env = Mock()
        obs = GymObservation(env=env)

        assert not isinstance(obs, torch.Tensor)


class TestGymObservationWithRealAttributes:
    """Test GymObservation with different environment attributes."""

    def test_with_env_having_gym_id(self) -> None:
        """Test with an environment that has _gym_id attribute."""
        env = Mock()
        env._gym_id = "PushT-v0"
        env.max_episode_steps = 100

        obs = GymObservation(env=env, episode_id=10)

        assert obs.env._gym_id == "PushT-v0"  # type: ignore[attr-defined]
        assert obs.env.max_episode_steps == 100  # type: ignore[attr-defined]
        assert obs.episode_id == 10

    def test_seed_propagation(self) -> None:
        """Test that seed can be set and retrieved."""
        env = Mock()
        seed = 42

        obs = GymObservation(env=env, seed=seed)

        assert obs.seed == seed

    def test_max_steps_override(self) -> None:
        """Test that max_steps can override default."""
        env = Mock()
        env.max_episode_steps = 1000
        custom_max_steps = 500

        obs = GymObservation(env=env, max_steps=custom_max_steps)

        assert obs.max_steps == custom_max_steps
        # Original env attribute should be unchanged
        assert env.max_episode_steps == 1000  # type: ignore[attr-defined]

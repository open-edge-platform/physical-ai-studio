# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - LIBERO Gym."""

import numpy as np
import pytest
import torch

from getiaction.data.observation import Observation

# Import will fail gracefully if libero is not installed
pytest.importorskip("libero")
pytest.importorskip("robosuite")

from getiaction.gyms.libero import TASK_SUITE_MAX_STEPS, LiberoGym, create_libero_gyms


class TestLiberoGym:
    """Test LiberoGym core functionality."""

    @pytest.fixture
    def gym(self):
        """Create a LiberoGym instance for testing."""
        gym_instance = LiberoGym(
            task_suite="libero_spatial",
            task_id=0,
            observation_height=224,
            observation_width=224,
            init_states=False,  # Don't require init_states files in unit tests
        )
        yield gym_instance
        gym_instance.close()

    def test_initialization(self, gym):
        """Test basic initialization and configuration."""
        assert gym.task_suite_name == "libero_spatial"
        assert gym.task_id == 0
        assert gym.observation_height == 224
        assert gym.observation_width == 224
        assert gym.control_mode == "relative"

    def test_spaces(self, gym):
        """Test observation and action spaces."""
        # Observation space
        obs_space = gym.observation_space
        assert "pixels" in obs_space.spaces
        assert obs_space.spaces["pixels"]["image"].shape == (224, 224, 3)
        assert "agent_pos" in obs_space.spaces
        assert obs_space.spaces["agent_pos"].shape == (8,)

        # Action space
        assert gym.action_space.shape == (7,)

    def test_reset_and_step(self, gym):
        """Test reset and step methods."""
        obs, info = gym.reset(seed=42)

        # Check reset output - now returns Observation
        assert isinstance(obs, Observation)
        assert obs.images["image"].shape == (1, 3, 224, 224)
        assert obs.state.shape == (1, 8)
        assert "task" in info
        assert "task_id" in info

        # Check step output
        action = gym.action_space.sample()
        obs, reward, terminated, truncated, info = gym.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert "is_success" in info

    def test_to_observation(self, gym):
        """Test conversion to Observation dataclass."""
        gym.reset(seed=42)
        obs, _, _, _, _ = gym.step(gym.action_space.sample())

        # obs is already an Observation from step()
        assert isinstance(obs, Observation)
        assert obs.images["image"].shape == (1, 3, 224, 224)
        assert obs.state.shape == (1, 8)
        assert obs.images["image"].dtype == torch.float32
        assert 0 <= obs.images["image"].max() <= 1.0

    @pytest.mark.parametrize("obs_type", ["pixels", "pixels_agent_pos"])
    def test_obs_types(self, obs_type):
        """Test different observation types."""
        gym_instance = LiberoGym(task_suite="libero_spatial", task_id=0, obs_type=obs_type, init_states=False)
        obs, _ = gym_instance.reset(seed=42)

        # obs is now an Observation object
        assert isinstance(obs, Observation)
        assert "image" in obs.images
        if obs_type == "pixels_agent_pos":
            assert obs.state is not None
        else:
            assert obs.state is None

        gym_instance.close()

    def test_different_suite(self):
        """Test different task suite."""
        gym_instance = LiberoGym(task_suite="libero_object", task_id=0, init_states=False)
        assert gym_instance.get_max_episode_steps() == TASK_SUITE_MAX_STEPS["libero_object"]
        assert gym_instance.max_episode_steps == TASK_SUITE_MAX_STEPS["libero_object"]
        gym_instance.close()

    def test_create_libero_gyms(self):
        """Test gym creation helper."""
        gyms = create_libero_gyms(
            task_suites=["libero_spatial", "libero_object"],
            task_ids=[0],
            init_states=False,  # Don't require init_states files in unit tests
        )

        assert len(gyms) == 2
        assert {g.task_suite_name for g in gyms} == {"libero_spatial", "libero_object"}

        for g in gyms:
            g.close()

    def test_invalid_suite(self):
        """Test error handling for invalid suite."""
        with pytest.raises(ValueError, match="Unknown LIBERO suite"):
            LiberoGym(task_suite="invalid_suite", task_id=0)

    def test_custom_size(self):
        """Test custom observation size."""
        gym_instance = LiberoGym(
            task_suite="libero_spatial",
            task_id=0,
            observation_height=128,
            observation_width=128,
            init_states=False,
        )
        obs, _ = gym_instance.reset(seed=42)
        # Observation has shape (1, 3, H, W)
        assert obs.images["image"].shape == (1, 3, 128, 128)
        gym_instance.close()

    def test_render(self, gym):
        """Test render method."""
        gym.reset(seed=42)
        image = gym.render()
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3

    def test_close_idempotent(self):
        """Test close can be called multiple times."""
        gym_instance = LiberoGym(task_suite="libero_spatial", task_id=0, init_states=False)
        gym_instance.close()
        gym_instance.close()  # Should not raise

    def test_control_mode(self):
        """Test control mode configuration."""
        # Test relative mode (default)
        gym_rel = LiberoGym(task_suite="libero_spatial", task_id=0, control_mode="relative", init_states=False)
        gym_rel.reset(seed=42)
        gym_rel.close()

        # Test absolute mode
        gym_abs = LiberoGym(task_suite="libero_spatial", task_id=0, control_mode="absolute", init_states=False)
        gym_abs.reset(seed=42)
        gym_abs.close()

    def test_check_success(self, gym):
        """Test success checking functionality."""
        gym.reset(seed=42)
        success = gym.check_success()
        assert isinstance(success, bool)

    def test_action_validation(self, gym):
        """Test action shape validation."""
        gym.reset(seed=42)

        # Invalid 2D action should raise
        with pytest.raises(ValueError, match="Expected 1-D action"):
            gym.step(np.zeros((2, 7)))

    def test_sample_action(self, gym):
        """Test sample_action method."""
        action = gym.sample_action()
        assert isinstance(action, torch.Tensor)
        assert action.shape == (7,)
        assert action.dtype == torch.float32

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LiberoGym with policies.

These tests verify that LiberoGym works correctly with our LeRobot policy wrappers
in a full evaluation loop: gym -> observation -> policy -> action -> gym.
"""

import pytest
import torch

# Skip if LIBERO is not installed
pytest.importorskip("libero")
pytest.importorskip("robosuite")

from getiaction.data.observation import Observation
from getiaction.gyms.libero import LiberoGym, create_libero_gyms
from getiaction.policies.lerobot import LeRobotPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


@pytest.fixture
def gym():
    """Create a LiberoGym instance for testing."""
    gym_instance = LiberoGym(
        task_suite="libero_spatial",
        task_id=0,
        observation_height=256,
        observation_width=256,
    )
    yield gym_instance
    gym_instance.close()


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def policy_features():
    """Create feature definitions matching LiberoGym output."""
    input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return input_features, output_features


class TestLiberoGymEndToEnd:
    """End-to-end integration tests for LiberoGym."""

    def test_gym_to_observation_format(self, gym, device):
        """Test that gym output is directly usable by policies."""
        obs, info = gym.reset(seed=42)

        # Verify observation format before moving to device
        assert isinstance(obs, Observation)
        assert "image" in obs.images
        assert "image2" in obs.images

        # Move to device and verify shapes/device
        # LiberoGym returns torch tensors, so use .to() not .to_torch()
        obs = obs.to(device)
        assert obs.images["image"].shape == (1, 3, 256, 256)
        assert obs.state.shape == (1, 8)
        assert str(obs.images["image"].device).startswith(device)

    def test_policy_inference_from_gym_observation(self, gym, device, policy_features):
        """Test that policy can process gym observations and produce valid actions."""
        input_features, output_features = policy_features

        # Create policy
        policy = LeRobotPolicy(
            policy_name="diffusion",
            input_features=input_features,
            output_features=output_features,
            config_kwargs={"crop_shape": None},
        )
        policy.to(device)
        policy.eval()

        # Get observation from gym
        obs, info = gym.reset(seed=42)
        obs = obs.to(device)

        # Run policy inference
        with torch.no_grad():
            action = policy.select_action(obs)

        # Verify action format
        assert isinstance(action, torch.Tensor)
        assert action.shape == (1, 7)  # Batch size 1, action dim 7
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_full_rollout_loop(self, gym, device, policy_features):
        """Test a complete rollout loop: gym -> policy -> gym -> policy -> ..."""
        input_features, output_features = policy_features

        # Create policy
        policy = LeRobotPolicy(
            policy_name="diffusion",
            input_features=input_features,
            output_features=output_features,
            config_kwargs={"crop_shape": None},
        )
        policy.to(device)
        policy.eval()

        # Reset
        obs, info = gym.reset(seed=42)
        obs = obs.to(device)
        policy.reset()

        # Run rollout
        num_steps = 10
        total_reward = 0.0

        for step in range(num_steps):
            # Policy inference
            with torch.no_grad():
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = gym.step(action.squeeze(0).cpu().numpy())
            obs = obs.to(device)
            total_reward += reward

            # Verify step outputs
            assert isinstance(obs, Observation)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "is_success" in info

            if terminated:
                break

        # Verify we completed the loop
        assert step >= 0

    def test_multiple_episodes(self, gym, device, policy_features):
        """Test running multiple episodes with reset."""
        input_features, output_features = policy_features

        policy = LeRobotPolicy(
            policy_name="diffusion",
            input_features=input_features,
            output_features=output_features,
            config_kwargs={"crop_shape": None},
        )
        policy.to(device)
        policy.eval()

        num_episodes = 2
        steps_per_episode = 5

        for ep in range(num_episodes):
            obs, info = gym.reset(seed=42 + ep)
            obs = obs.to(device)
            policy.reset()

            for step in range(steps_per_episode):
                with torch.no_grad():
                    action = policy.select_action(obs)
                obs, reward, terminated, truncated, info = gym.step(action.squeeze(0).cpu().numpy())
                obs = obs.to(device)

                if terminated:
                    break

    def test_create_libero_gyms_with_policy(self, device, policy_features):
        """Test create_libero_gyms helper works with policy evaluation."""
        input_features, output_features = policy_features

        # Create multiple gyms
        gyms = create_libero_gyms(
            task_suites=["libero_spatial", "libero_object"],
            task_ids=[0],
            observation_height=256,
            observation_width=256,
        )

        assert len(gyms) == 2

        # Create policy
        policy = LeRobotPolicy(
            policy_name="diffusion",
            input_features=input_features,
            output_features=output_features,
            config_kwargs={"crop_shape": None},
        )
        policy.to(device)
        policy.eval()

        # Test each gym
        for g in gyms:
            obs, info = g.reset(seed=42)
            obs = obs.to(device)
            policy.reset()

            with torch.no_grad():
                action = policy.select_action(obs)

            obs, reward, terminated, truncated, info = g.step(action.squeeze(0).cpu().numpy())

            assert isinstance(obs, Observation)
            g.close()

    def test_control_modes_with_policy(self, device, policy_features):
        """Test both control modes work with policy evaluation."""
        input_features, output_features = policy_features

        policy = LeRobotPolicy(
            policy_name="diffusion",
            input_features=input_features,
            output_features=output_features,
            config_kwargs={"crop_shape": None},
        )
        policy.to(device)
        policy.eval()

        for control_mode in ["relative", "absolute"]:
            gym = LiberoGym(
                task_suite="libero_spatial",
                task_id=0,
                observation_height=256,
                observation_width=256,
                control_mode=control_mode,
            )

            obs, info = gym.reset(seed=42)
            obs = obs.to(device)
            policy.reset()

            with torch.no_grad():
                action = policy.select_action(obs)

            obs, reward, terminated, truncated, info = gym.step(action.squeeze(0).cpu().numpy())

            assert isinstance(obs, Observation)
            gym.close()

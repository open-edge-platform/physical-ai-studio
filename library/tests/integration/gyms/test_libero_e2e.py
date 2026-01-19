# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for LiberoGym with first-party ACT policy.

These tests verify that LiberoGym works correctly with our first-party ACT policy
in a full evaluation loop: gym -> observation -> policy -> action -> gym.
"""

import pytest
import torch

# Skip if LIBERO is not installed
pytest.importorskip("libero")
pytest.importorskip("robosuite")

from getiaction.data import Feature, FeatureType, Observation
from getiaction.gyms.libero import LiberoGym, create_libero_gyms
from getiaction.policies import ACT, ACTConfig
from getiaction.policies.act.model import ACT as ACTModel
from getiaction.policies.utils.normalization import NormalizationParameters


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
def act_policy(device):
    """Create a first-party ACT policy matching LiberoGym output."""
    input_features = {
        "image": Feature(
            ftype=FeatureType.VISUAL,
            shape=(3, 256, 256),
            normalization_data=NormalizationParameters(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ),
        "image2": Feature(
            ftype=FeatureType.VISUAL,
            shape=(3, 256, 256),
            normalization_data=NormalizationParameters(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ),
        "state": Feature(
            ftype=FeatureType.STATE,
            shape=(8,),
            normalization_data=NormalizationParameters(mean=[0.0] * 8, std=[1.0] * 8),
        ),
    }
    output_features = {
        "action": Feature(
            ftype=FeatureType.ACTION,
            shape=(7,),
            normalization_data=NormalizationParameters(mean=[0.0] * 7, std=[1.0] * 7),
        ),
    }

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=100,
        dim_model=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
    )
    model = ACTModel.from_config(config)
    policy = ACT(model)
    policy.to(device)
    policy.eval()
    return policy


class TestLiberoGymEndToEnd:
    """End-to-end integration tests for LiberoGym with first-party ACT."""

    def test_gym_to_observation_format(self, gym, device):
        """Test that gym output is directly usable by policies."""
        obs, info = gym.reset(seed=42)

        # Verify observation format before moving to device
        assert isinstance(obs, Observation)
        assert "image" in obs.images
        assert "image2" in obs.images

        # Move to device and verify shapes/device
        obs = obs.to(device)
        assert obs.images["image"].shape == (1, 3, 256, 256)
        assert obs.state.shape == (1, 8)
        assert str(obs.images["image"].device).startswith(device)

    def test_policy_inference_from_gym_observation(self, gym, device, act_policy):
        """Test that ACT policy can process gym observations and produce valid actions."""
        # Get observation from gym
        obs, info = gym.reset(seed=42)
        obs = obs.to(device)

        # Run policy inference
        with torch.no_grad():
            action = act_policy.select_action(obs)

        # Verify action format (select_action returns single action, not chunk)
        assert isinstance(action, torch.Tensor)
        assert action.shape[0] == 1  # Batch size 1
        assert action.shape[1] == 7  # Action dim (single action)

    def test_full_rollout_loop(self, gym, device, act_policy):
        """Test a complete rollout loop: gym -> policy -> gym -> policy -> ..."""
        # Reset
        obs, info = gym.reset(seed=42)
        obs = obs.to(device)
        act_policy.reset()

        # Run rollout
        num_steps = 10
        total_reward = 0.0

        for step in range(num_steps):
            # Policy inference
            with torch.no_grad():
                action = act_policy.select_action(obs)

            # select_action returns single action [batch, action_dim]
            action_to_execute = action.squeeze(0).cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = gym.step(action_to_execute)
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

    def test_multiple_episodes(self, gym, device, act_policy):
        """Test running multiple episodes with reset."""
        num_episodes = 2
        steps_per_episode = 5

        for ep in range(num_episodes):
            obs, info = gym.reset(seed=42 + ep)
            obs = obs.to(device)
            act_policy.reset()

            for step in range(steps_per_episode):
                with torch.no_grad():
                    action = act_policy.select_action(obs)
                # select_action returns single action [batch, action_dim]
                action_to_execute = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, info = gym.step(action_to_execute)
                obs = obs.to(device)

                if terminated:
                    break

    def test_create_libero_gyms_with_policy(self, device, act_policy):
        """Test create_libero_gyms helper works with ACT policy evaluation."""
        # Create multiple gyms
        gyms = create_libero_gyms(
            task_suites=["libero_spatial", "libero_object"],
            task_ids=[0],
            observation_height=256,
            observation_width=256,
        )

        assert len(gyms) == 2

        # Test each gym
        for g in gyms:
            obs, info = g.reset(seed=42)
            obs = obs.to(device)
            act_policy.reset()

            with torch.no_grad():
                action = act_policy.select_action(obs)

            # select_action returns single action [batch, action_dim]
            action_to_execute = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = g.step(action_to_execute)

            assert isinstance(obs, Observation)
            g.close()

    def test_control_modes_with_policy(self, device, act_policy):
        """Test both control modes work with ACT policy evaluation."""
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
            act_policy.reset()

            with torch.no_grad():
                action = act_policy.select_action(obs)

            # select_action returns single action [batch, action_dim]
            action_to_execute = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = gym.step(action_to_execute)

            assert isinstance(obs, Observation)
            gym.close()

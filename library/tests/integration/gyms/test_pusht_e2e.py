# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for PushTGym with first-party ACT policy.

These tests verify that PushTGym works correctly with our first-party ACT policy
in a full evaluation loop: gym -> observation -> policy -> action -> gym.
"""

import pytest
import torch

# Skip if gym_pusht is not installed
pytest.importorskip("gym_pusht")

from physicalai.data import FeatureType, Observation
from physicalai.gyms.pusht import PushTGym
from physicalai.policies import ACT


@pytest.fixture
def gym():
    """Create a PushTGym instance for testing."""
    gym_instance = PushTGym()
    yield gym_instance
    gym_instance.close()


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def act_policy(device):
    """Create a first-party ACT policy matching PushTGym output."""
    dataset_stats = {
        "top": {
            "name": "top",
            "type": FeatureType.VISUAL,
            "shape": (3, 96, 96),
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "state": {
            "name": "state",
            "type": FeatureType.STATE,
            "shape": (2,),
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
        },
        "action": {
            "name": "action",
            "type": FeatureType.ACTION,
            "shape": (2,),
            "mean": [0.0, 0.0],
            "std": [1.0, 1.0],
        },
    }
    policy = ACT(
        chunk_size=100,
        dim_model=256,
        n_encoder_layers=2,
        n_decoder_layers=1,
        dataset_stats=dataset_stats,
    )
    policy.to(device)
    policy.eval()
    return policy


class TestPushTGymEndToEnd:
    """End-to-end integration tests for PushTGym with first-party ACT."""

    def test_gym_to_observation_format(self, gym, device):
        """Test that gym output is directly usable by policies."""
        obs, info = gym.reset(seed=42)

        # Verify observation format before moving to device
        assert isinstance(obs, Observation)
        assert obs.images is not None
        assert obs.state is not None

        # Move to device and verify shapes/device
        obs = obs.to(device)
        assert obs.images.shape == (1, 3, 96, 96)
        assert obs.state.shape == (1, 2)
        assert str(obs.images.device).startswith(device)

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
        assert action.shape[1] == 2  # Action dim (x, y)

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
            assert "coverage" in info

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
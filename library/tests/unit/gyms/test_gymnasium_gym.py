"""Test gymnasium wrapper."""
import pytest
import torch
from getiaction.gyms import GymnasiumGym
from getiaction.data.observation import Observation

from .base import BaseTestGym     

class TestGymnasiumGym(BaseTestGym):
    """Tests the GymnasiumGym adapter using BaseTestGym standards."""

    def setup_env(self):
        # Simple default environment
        self.env = GymnasiumGym(gym_id="CartPole-v1")


def test_sample_action_always_batch_dim():
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    a = env.sample_action()

    assert isinstance(a, torch.Tensor)
    assert a.ndim == 2  # Always [B,Dim]
    assert a.shape[0] == 1  # B for non-vector = 1
    assert a.shape[1] == 1 or a.shape[1] > 1  # Dim may be 1 or more

    env.close()


def test_reset_returns_observation_and_info():
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    obs, info = env.reset()

    assert isinstance(obs, Observation)
    assert obs.batch_size == 1

    assert isinstance(info, dict)
    env.close()


def test_step_returns_batched_elements():
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    obs, _ = env.reset()

    action = env.sample_action()
    assert action.ndim == 2                       # 🔥 [1,Dim]

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, Observation)
    assert obs.batch_size == 1                    # 🔥 Still batched output

    assert isinstance(reward, float) or torch.is_tensor(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


# ============================================================
# Vectorized mode — B must equal num_envs and Dim preserved
# ============================================================

@pytest.mark.parametrize("num_envs", [2, 4])
def test_vectorized_env_batch_shape_consistency(num_envs):
    env = GymnasiumGym.vectorize("CartPole-v1", num_envs=num_envs)
    obs, info = env.reset()

    assert isinstance(obs, Observation)
    assert obs.batch_size == num_envs                 # 🔥 Batch preserved

    action = env.sample_action()
    assert action.ndim == 2                           # 🔥 Always [B,Dim]
    assert action.shape[0] == num_envs

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, Observation)
    assert obs.batch_size == num_envs
    assert len(reward) == num_envs                    # Rewards are batched
    assert len(terminated) == num_envs
    assert len(truncated) == num_envs

    env.close()
"""Test gymnasium wrapper."""
import pytest
import torch
from physicalai.gyms import GymnasiumGym
from physicalai.data.observation import Observation

from .base import BaseTestGym


class TestGymnasiumGym(BaseTestGym):
    """Tests the GymnasiumGym adapter using BaseTestGym standards."""

    def setup_env(self):
        # Simple default environment
        self.env = GymnasiumGym(gym_id="CartPole-v1")


def test_sample_action_always_batch_dim():
    """Sampled actions must always be batched [B,Dim]."""
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    a = env.sample_action()

    assert isinstance(a, torch.Tensor)
    assert a.ndim == 2  # always [B,Dim]
    assert a.shape[0] == 1  # B = 1 for non-vectorized envs
    assert a.shape[1] >= 1  # dim must exist even for discrete

    env.close()


def test_reset_returns_observation_and_info():
    """Reset must return a batched Observation and dict info."""
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    obs, info = env.reset()

    assert isinstance(obs, Observation)
    assert obs.batch_size == 1  #  always batched for single env
    assert isinstance(info, dict)

    env.close()


def test_step_returns_batched_elements():
    """Step must consume [B,Dim] action and return batched values."""
    env = GymnasiumGym(gym_id="CartPole-v1", render_mode=None)
    obs, _ = env.reset()

    action = env.sample_action()
    assert action.ndim == 2  #  always [1,Dim]

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, Observation)
    assert obs.batch_size == 1  #  remains batched

    assert isinstance(reward, (float, torch.Tensor))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


@pytest.mark.parametrize("num_envs", [2, 4])
def test_vectorized_env_batch_shape_consistency(num_envs):
    """Vector env must respect batch size and preserve Dim."""
    env = GymnasiumGym.vectorize("CartPole-v1", num_envs=num_envs)
    obs, info = env.reset()

    assert isinstance(obs, Observation)
    assert obs.batch_size == num_envs  #  batch preserved

    action = env.sample_action()
    assert action.ndim == 2  #  always [B,Dim]
    assert action.shape[0] == num_envs  #  B must match vector count

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, Observation)
    assert obs.batch_size == num_envs
    assert len(reward) == num_envs  #  batched rewards
    assert len(terminated) == num_envs
    assert len(truncated) == num_envs

    env.close()

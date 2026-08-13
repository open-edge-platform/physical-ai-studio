# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rollout functionality."""

from __future__ import annotations

import pytest
import torch

from physicalai.gyms import PushTGym, GymnasiumGym
from physicalai.eval import rollout


@pytest.fixture
def env_pusht():
    """PushT Gym fixture."""
    return PushTGym()


@pytest.fixture
def env_cartpole():
    """CartPole env fixture."""
    return GymnasiumGym("CartPole-v1")


@pytest.fixture
def env_cartpole_vec():
    """Vectorized CartPole env fixture."""
    return GymnasiumGym.vectorize("CartPole-v1", num_envs=3)


def _policy_from_env(env, dummy_policy):
    """Create a dummy policy matching an environment's action space."""
    action = env.sample_action()
    assert action.ndim == 2
    action_shape = tuple(action.shape[1:])
    action_dtype = action.dtype
    action_max = 1 if action_dtype in (torch.int64, torch.int32) else None
    action_min = 0 if action_dtype in (torch.int64, torch.int32) else None
    return dummy_policy.create(
        action_shape=action_shape,
        action_dtype=action_dtype,
        action_max=action_max,
        action_min=action_min,
    )


class TestRollout:
    """Tests for rollout with dynamic action shape."""

    @pytest.mark.parametrize(
        "env_fixture, policy_env_fixture",
        [
            ("env_pusht", "env_pusht"),
            ("env_cartpole", "env_cartpole"),
            ("env_cartpole_vec", "env_cartpole_vec"),
        ]
    )
    def test_rollout_executes_successfully(self, request, dummy_policy, env_fixture, policy_env_fixture):
        env = request.getfixturevalue(env_fixture)
        policy_env = request.getfixturevalue(policy_env_fixture)

        policy = _policy_from_env(policy_env, dummy_policy)

        result = rollout(env=env, policy=policy, seed=42, max_steps=5, return_observations=False)

        assert "episode_length" in result
        assert "sum_reward" in result
        assert "max_reward" in result

    @pytest.mark.parametrize(
        "env_fixture",
        ["env_pusht", "env_cartpole", "env_cartpole_vec"],
    )
    def test_rollout_return_types(self, request, dummy_policy, env_fixture):
        """Rollout returns correct types."""
        env = request.getfixturevalue(env_fixture)

        policy = _policy_from_env(env, dummy_policy)

        result = rollout(env=env, policy=policy, seed=42, max_steps=5, return_observations=False)

        assert isinstance(result["episode_length"], int)
        assert isinstance(result["sum_reward"], (float, torch.Tensor))


# ============================================================================ #
# _collect_frame Tests                                                         #
# ============================================================================ #


class TestCollectFrame:
    """Tests for _collect_frame handling of different image types."""

    def test_collect_frame_tensor_single_camera(self) -> None:
        """Test _collect_frame with a single-camera Tensor observation."""
        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        # Single camera: images is a Tensor [1, C, H, W]
        images = torch.rand(1, 3, 96, 96)
        obs = Observation(state=torch.randn(1, 4), images=images)
        frame = _collect_frame(obs, "camera")

        assert frame is not None
        assert frame.shape == (96, 96, 3)  # H, W, C after permute

    def test_collect_frame_ndarray_single_camera(self) -> None:
        """Test _collect_frame with a single-camera numpy ndarray observation."""
        import numpy as np

        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        images = np.random.rand(1, 3, 96, 96).astype(np.float32)
        obs = Observation(state=torch.randn(1, 4), images=images)
        frame = _collect_frame(obs, "camera")

        assert frame is not None
        assert frame.shape == (96, 96, 3)

    def test_collect_frame_dict_multi_camera(self) -> None:
        """Test _collect_frame with a dict of images (multiple cameras)."""
        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        images = {
            "top": torch.rand(1, 3, 64, 64),
            "wrist": torch.rand(1, 3, 64, 64),
        }
        obs = Observation(state=torch.randn(1, 4), images=images)

        frame = _collect_frame(obs, "top")
        assert frame is not None
        assert frame.shape == (64, 64, 3)

    def test_collect_frame_dict_missing_key(self) -> None:
        """Test _collect_frame returns None for missing key in dict."""
        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        images = {"top": torch.rand(1, 3, 64, 64)}
        obs = Observation(state=torch.randn(1, 4), images=images)

        frame = _collect_frame(obs, "nonexistent")
        assert frame is None

    def test_collect_frame_no_images(self) -> None:
        """Test _collect_frame returns None when images is None."""
        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        obs = Observation(state=torch.randn(1, 4), images=None)
        frame = _collect_frame(obs, "camera")
        assert frame is None

    def test_collect_frame_unsupported_type(self) -> None:
        """Test _collect_frame returns None for unsupported image type."""
        from physicalai.data import Observation
        from physicalai.eval.rollout.functional import _collect_frame

        obs = Observation(state=torch.randn(1, 4), images="not_an_image")
        frame = _collect_frame(obs, "camera")
        assert frame is None


# ============================================================================ #
# episode_index / rollout_idx forwarding Tests                                 #
# ============================================================================ #


class _SpyGym:
    """Minimal Gym that records the kwargs passed to ``reset``.

    Terminates every episode after a single step so rollouts stay tiny.
    """

    def __init__(self, batch_size: int = 1, action_dim: int = 2) -> None:
        self.reset_calls: list[dict] = []
        self.max_steps = 100
        self._batch_size = batch_size
        self._action_dim = action_dim

    def _obs(self):
        from physicalai.data import Observation

        return Observation(state=torch.zeros(self._batch_size, 4))

    def reset(self, *, seed=None, episode_index=0, **_kwargs):
        self.reset_calls.append({"seed": seed, "episode_index": episode_index})
        return self._obs(), {}

    def step(self, _action):
        reward = torch.zeros(self._batch_size)
        terminated = torch.ones(self._batch_size, dtype=torch.bool)  # end after 1 step
        truncated = torch.zeros(self._batch_size, dtype=torch.bool)
        info = {"is_success": True}
        return self._obs(), reward, terminated, truncated, info

    def close(self) -> None:
        pass

    def sample_action(self) -> torch.Tensor:
        return torch.zeros(self._batch_size, self._action_dim)

    def to_observation(self, raw_obs):
        return raw_obs


class TestEpisodeIndexForwarding:
    """Tests that rollout_idx flows through as env.reset(episode_index=...)."""

    def test_setup_rollout_forwards_rollout_idx(self, dummy_policy) -> None:
        """setup_rollout passes rollout_idx as episode_index to env.reset."""
        from physicalai.eval.rollout.functional import setup_rollout

        env = _SpyGym()
        setup_rollout(env, dummy_policy, seed=1, max_steps=5, rollout_idx=7)

        assert env.reset_calls == [{"seed": 1, "episode_index": 7}]

    def test_rollout_forwards_rollout_idx(self, dummy_policy) -> None:
        """rollout forwards rollout_idx down to env.reset."""
        from physicalai.eval import rollout

        env = _SpyGym()
        rollout(env=env, policy=dummy_policy, seed=0, max_steps=1, rollout_idx=3)

        assert env.reset_calls[0]["episode_index"] == 3

    def test_rollout_default_rollout_idx_is_zero(self, dummy_policy) -> None:
        """rollout defaults episode_index to 0 when rollout_idx is omitted."""
        from physicalai.eval import rollout

        env = _SpyGym()
        rollout(env=env, policy=dummy_policy, seed=0, max_steps=1)

        assert env.reset_calls[0]["episode_index"] == 0

    def test_evaluate_policy_increments_episode_index(self, dummy_policy) -> None:
        """evaluate_policy resets with episode_index 0, 1, 2 across episodes."""
        from physicalai.eval.rollout.functional import evaluate_policy

        env = _SpyGym()
        evaluate_policy(env, dummy_policy, n_episodes=3, start_seed=100, max_steps=1)

        episode_indices = [call["episode_index"] for call in env.reset_calls]
        assert episode_indices == [0, 1, 2]

    def test_evaluate_policy_episode_index_aligned_with_seed(self, dummy_policy) -> None:
        """episode_index stays aligned with seed offset (guards off-by-one)."""
        from physicalai.eval.rollout.functional import evaluate_policy

        env = _SpyGym()
        start_seed = 100
        evaluate_policy(env, dummy_policy, n_episodes=3, start_seed=start_seed, max_steps=1)

        for call in env.reset_calls:
            assert call["episode_index"] == call["seed"] - start_seed

    def test_evaluate_policy_episode_index_without_seed(self, dummy_policy) -> None:
        """episode_index still increments when start_seed is None."""
        from physicalai.eval.rollout.functional import evaluate_policy

        env = _SpyGym()
        evaluate_policy(env, dummy_policy, n_episodes=3, start_seed=None, max_steps=1)

        episode_indices = [call["episode_index"] for call in env.reset_calls]
        seeds = [call["seed"] for call in env.reset_calls]
        assert episode_indices == [0, 1, 2]
        assert seeds == [None, None, None]

    def test_evaluate_policy_vectorized_indexes_rollouts_not_episodes(self, dummy_policy) -> None:
        """With a batched env one rollout covers all episodes -> single reset at index 0."""
        from physicalai.eval.rollout.functional import evaluate_policy

        env = _SpyGym(batch_size=3)
        evaluate_policy(env, dummy_policy, n_episodes=3, start_seed=0, max_steps=1)

        episode_indices = [call["episode_index"] for call in env.reset_calls]
        assert episode_indices == [0]

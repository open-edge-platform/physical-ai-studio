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

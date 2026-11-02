# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for ``RoboCasaGym``.

Spins up MuJoCo, so it is marked ``integration`` + ``slow`` and runs only
in the heavy CI lane. Use ``task="CloseFridge"`` — the cheapest atomic
scene verified by the §7.4 Step 2 standalone check.

These tests assume a dedicated ``.venv-robocasa`` set up via
``library/scripts/benchmark/install_robocasa.sh`` with downloaded kitchen assets,
and ``MUJOCO_GL=egl`` exported on the runner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("robocasa")
pytest.importorskip("robosuite")

from physicalai.data import Observation  # noqa: E402
from physicalai.gyms.robocasa import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_CAMERAS,
    OBS_STATE_DIM,
    RoboCasaGym,
)

@pytest.fixture
def gym():
    """Single-task RoboCasaGym at the smallest sane resolution."""
    gym_instance = RoboCasaGym(
        task="CloseFridge",
        observation_height=128,
        observation_width=128,
    )
    yield gym_instance
    gym_instance.close()


class TestRoboCasaGymEndToEnd:
    """End-to-end smoke tests for ``RoboCasaGym``."""

    def test_reset_returns_observation_shape(self, gym):
        """``reset`` returns an ``Observation`` with the documented shapes."""
        obs, info = gym.reset(seed=0)

        assert isinstance(obs, Observation)
        assert isinstance(info, dict)
        assert info["task"] == "CloseFridge"
        assert "is_success" in info

        assert isinstance(obs.images, dict)
        assert set(obs.images.keys()) == set(DEFAULT_CAMERAS)
        for cam in DEFAULT_CAMERAS:
            assert obs.images[cam].shape == (1, 3, 128, 128)
            assert obs.images[cam].dtype == torch.float32
            assert 0.0 <= float(obs.images[cam].max()) <= 1.0

        assert obs.state is not None
        assert obs.state.shape == (1, OBS_STATE_DIM)
        assert obs.state.dtype == torch.float32

        assert obs.task is not None
        assert isinstance(obs.task, list)
        assert len(obs.task) == 1
        assert isinstance(obs.task[0], str)

    def test_step_random_action(self, gym):
        """One ``step(sample_action())`` returns the gym-API 5-tuple."""
        gym.reset(seed=0)

        action = gym.sample_action()
        assert action.shape == (ACTION_DIM,)

        obs, reward, terminated, truncated, info = gym.step(action)

        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "is_success" in info

    def test_step_accepts_numpy_action(self, gym):
        """``step`` also accepts a numpy array (no torch conversion needed)."""
        gym.reset(seed=0)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        obs, reward, terminated, truncated, info = gym.step(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, float)

    def test_step_rejects_2d_action(self, gym):
        """A 2-D action raises ``ValueError``."""
        gym.reset(seed=0)
        with pytest.raises(ValueError, match="Expected 1-D action"):
            gym.step(np.zeros((2, ACTION_DIM), dtype=np.float32))

    def test_episode_termination_info_keys(self, gym):
        """``info`` carries the keys the benchmark loop reads each step."""
        gym.reset(seed=0)
        _, _, _, _, info = gym.step(gym.sample_action())
        # Benchmark loop reads these keys directly; verify presence.
        assert "is_success" in info
        assert "done" in info
        assert "task" in info
        assert info["task"] == "CloseFridge"

    def test_short_random_rollout(self, gym):
        """A short random rollout completes without NaNs in observations."""
        obs, _ = gym.reset(seed=0)
        num_steps = 5
        for _ in range(num_steps):
            action = gym.sample_action()
            obs, _, terminated, truncated, _ = gym.step(action)
            assert isinstance(obs, Observation)
            for cam, img in obs.images.items():
                assert not torch.isnan(img).any(), f"NaN in image[{cam}]"
            assert obs.state is not None
            assert not torch.isnan(obs.state).any(), "NaN in state"
            if terminated or truncated:
                break

    def test_render(self, gym):
        """``render`` returns an HWC uint8 image after reset."""
        gym.reset(seed=0)
        frame = gym.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3

    def test_close_idempotent(self):
        """``close`` may be called twice without error."""
        gym_instance = RoboCasaGym(
            task="CloseFridge",
            observation_height=128,
            observation_width=128,
        )
        gym_instance.close()
        gym_instance.close()

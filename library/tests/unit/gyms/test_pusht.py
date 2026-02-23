# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - PushT Gym."""

import numpy as np
import torch

from physicalai.data.observation import Observation
from physicalai.gyms import PushTGym
from tests.unit.gyms.base import BaseTestGym


class TestPushTGym(BaseTestGym):
    """
    Tests the specific implementation of the PushTGym wrapper.
    """

    def setup_env(self):
        """Sets up the PushTGym environment for testing."""
        self.env = PushTGym()

    def test_pushtgym_default_parameters(self):
        """
        Tests if PushTGym initializes with the correct default parameters.
        """
        # The env is already created by the setup_env fixture
        assert self.env._env.spec.id == "gym_pusht/PushT-v0"

    def test_pushtgym_custom_parameters(self):
        """
        Tests if PushTGym can be initialized with custom parameters.
        """
        self.env.close()
        self.env = PushTGym(
            obs_type="state",
        )

    def test_convert_raw_observation_with_pixels_and_state(self):
        """Test PushTGym.convert_raw_observation() with typical PushT observation."""
        raw_obs = {
            "pixels": (np.random.rand(1, 480, 640, 3) * 255).astype(np.uint8),
            "agent_pos": np.array([[0.5, 0.3]], dtype=np.float32),
        }

        obs = PushTGym.convert_raw_to_observation(raw_obs)

        assert isinstance(obs, Observation)
        assert obs.images is not None
        assert obs.state is not None
        # PushT uses single camera, so images should be a direct tensor
        assert hasattr(obs.images, "shape")  # Should be tensor, not dict
        assert obs.images.shape == (
            1,
            3,
            480,
            640,
        )  # Batched, CHW  # type: ignore[attr-defined]
        assert obs.state.shape == (1, 2)  # Batched state  # type: ignore[attr-defined]
        assert obs.images.dtype == torch.float32  # type: ignore[attr-defined]
        assert torch.max(obs.images) <= 1.0  # type: ignore[attr-defined]

    def test_to_observation_delegates_to_static_method(self):
        """Test that instance method delegates to static method."""
        raw_obs = {
            "pixels": np.random.rand(64, 64, 3).astype(np.float32),
            "agent_pos": np.array([0.1, 0.2], dtype=np.float32),
        }

        # Both methods should produce same result
        static_result = PushTGym.convert_raw_to_observation(raw_obs)
        instance_result = self.env.to_observation(raw_obs)

        assert isinstance(static_result, Observation)
        assert isinstance(instance_result, Observation)
        assert static_result.images.shape == instance_result.images.shape  # type: ignore[attr-defined]
        assert static_result.state.shape == instance_result.state.shape  # type: ignore[attr-defined]

def test_state_only_obs():
    """Test we can convert with only position."""
    raw_obs = {"agent_pos": np.array([[0.4,0.7]],dtype=np.float32)}
    obs = PushTGym.convert_raw_to_observation(raw_obs)

    assert obs.images is None
    assert obs.state.shape == (1,2)

def test_pixels_only_obs():
    """Test that we can convert only pixels"""
    raw_obs = {"pixels": np.random.rand(1,32,32,3).astype(np.float32)}
    obs = PushTGym.convert_raw_to_observation(raw_obs)

    assert obs.images.shape == (1,3,32,32)
    assert obs.state is None

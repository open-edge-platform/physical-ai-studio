# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gym observation to Observation conversion."""

import numpy as np
import pytest
import torch

from getiaction.data.observation import Observation, gym_observation_to_observation


class TestGymObservationToObservation:
    """Tests for standalone gym_observation_to_observation function."""

    def test_converts_pixels_and_state(self):
        """Test conversion with both pixels and state - single camera returns direct tensor."""
        gym_obs = {
            "pixels": np.random.rand(480, 640, 3).astype(np.float32),
            "agent_pos": np.array([0.5, 0.3]),
        }

        obs = gym_observation_to_observation(gym_obs)

        # Single camera: images should be a direct tensor (not dict)
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 480, 640)
        assert obs.state is not None
        assert obs.state.shape == (1, 2)
        assert obs.images.dtype == torch.float32
        assert obs.state.dtype == torch.float32

    def test_converts_pixels_only(self):
        """Test conversion with only pixels - single camera returns direct tensor."""
        gym_obs = {"pixels": np.random.rand(224, 224, 3).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs)

        # Single camera: images should be a direct tensor
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 224, 224)
        assert obs.state is None

    def test_converts_state_only(self):
        """Test conversion with only state."""
        gym_obs = {"agent_pos": np.array([0.1, 0.2, 0.3, 0.4])}

        obs = gym_observation_to_observation(gym_obs)

        assert obs.images is None
        assert obs.state is not None
        assert obs.state.shape == (1, 4)

    def test_custom_camera_keys_multiple(self):
        """Test conversion with multiple camera keys - returns dict."""
        gym_obs = {"pixels": np.random.rand(100, 100, 3).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs, camera_keys=["front", "side"])

        # Multiple cameras: images should be a dict
        assert isinstance(obs.images, dict)
        assert "front" in obs.images
        assert "top" not in obs.images
        assert obs.images["front"].shape == (1, 3, 100, 100)

    def test_custom_camera_keys_single(self):
        """Test conversion with single custom camera key - returns direct tensor."""
        gym_obs = {"pixels": np.random.rand(100, 100, 3).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs, camera_keys=["front"])

        # Single camera: images should be a direct tensor
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 100, 100)

    def test_converts_uint8_to_float32(self):
        """Test conversion from uint8 images - single camera."""
        gym_obs = {"pixels": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}

        obs = gym_observation_to_observation(gym_obs)

        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.dtype == torch.float32

    def test_converts_float64_to_float32(self):
        """Test conversion from float64 - single camera."""
        gym_obs = {
            "pixels": np.random.rand(100, 100, 3).astype(np.float64),
            "agent_pos": np.array([0.5, 0.3], dtype=np.float64),
        }

        obs = gym_observation_to_observation(gym_obs)

        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.dtype == torch.float32
        assert obs.state is not None
        assert obs.state.dtype == torch.float32

    def test_handles_hwc_to_chw_conversion(self):
        """Test HWC to CHW image conversion - single camera."""
        # Create a known pattern to verify conversion
        pixels = np.zeros((2, 3, 3), dtype=np.float32)  # H=2, W=3, C=3
        pixels[:, :, 0] = 1.0  # Red channel

        gym_obs = {"pixels": pixels}
        obs = gym_observation_to_observation(gym_obs)

        # After conversion: (1, C=3, H=2, W=3)
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 2, 3)
        # Red channel should be first
        assert obs.images[0, 0, 0, 0] == 1.0

    def test_adds_batch_dimension(self):
        """Test that batch dimension is added - single camera."""
        gym_obs = {
            "pixels": np.random.rand(10, 10, 3).astype(np.float32),
            "agent_pos": np.array([0.5]),
        }

        obs = gym_observation_to_observation(gym_obs)

        # Check batch dimension is present
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.ndim == 4
        assert obs.images.shape[0] == 1
        assert obs.state is not None
        assert obs.state.ndim == 2
        assert obs.state.shape[0] == 1

    def test_handles_state_key_fallback(self):
        """Test fallback to 'state' key if 'agent_pos' not present."""
        gym_obs = {"state": np.array([0.1, 0.2, 0.3])}

        obs = gym_observation_to_observation(gym_obs)

        assert obs.state is not None
        assert obs.state.shape == (1, 3)

    def test_empty_observation(self):
        """Test conversion with empty observation."""
        gym_obs = {}

        obs = gym_observation_to_observation(gym_obs)

        assert obs.images is None
        assert obs.state is None


class TestObservationFromGymClassMethod:
    """Tests for Observation.from_gym class method."""

    def test_class_method_exists(self):
        """Test that from_gym class method exists."""
        assert hasattr(Observation, "from_gym")
        assert callable(Observation.from_gym)

    def test_class_method_produces_same_result_as_function(self):
        """Test that class method produces identical result to standalone function.

        Note: With default single camera, images is a direct tensor (not dict).
        """
        gym_obs = {
            "pixels": np.random.rand(100, 100, 3).astype(np.float32),
            "agent_pos": np.array([0.5, 0.3]),
        }

        obs1 = gym_observation_to_observation(gym_obs)
        obs2 = Observation.from_gym(gym_obs)

        # Single camera: images is a direct tensor
        assert isinstance(obs1.images, torch.Tensor)
        assert isinstance(obs2.images, torch.Tensor)
        assert torch.equal(obs1.images, obs2.images)
        assert torch.equal(obs1.state, obs2.state)

    def test_class_method_with_custom_camera_keys(self):
        """Test class method with custom camera keys.

        Note: Single camera still returns direct tensor, not dict.
        """
        gym_obs = {"pixels": np.random.rand(50, 50, 3).astype(np.float32)}

        obs = Observation.from_gym(gym_obs, camera_keys=["camera1"])

        # Single camera: images is a direct tensor
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 50, 50)

    def test_class_method_integration_with_observation_api(self):
        """Test that from_gym integrates well with other Observation methods."""
        gym_obs = {
            "pixels": np.random.rand(64, 64, 3).astype(np.float32),
            "agent_pos": np.array([0.1, 0.2]),
        }

        obs = Observation.from_gym(gym_obs)

        # Test to_dict()
        obs_dict = obs.to_dict()
        assert "images" in obs_dict
        assert "state" in obs_dict

        # Test keys()
        keys = obs.keys()
        assert "images" in keys
        assert "state" in keys

        # Test to(device)
        if torch.cuda.is_available():
            obs_cuda = obs.to("cuda")
            # Single camera: images is a direct tensor
            assert obs_cuda.images.device.type == "cuda"


class TestGymConversionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_grayscale_image(self):
        """Test conversion of grayscale (single channel) images.

        Note: Single camera returns direct tensor, not dict.
        """
        gym_obs = {"pixels": np.random.rand(100, 100, 1).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs)

        # Single camera: images is a direct tensor
        assert obs.images.shape == (1, 1, 100, 100)

    def test_rgba_image(self):
        """Test conversion of RGBA (4 channel) images.

        Note: Single camera returns direct tensor, not dict.
        """
        gym_obs = {"pixels": np.random.rand(100, 100, 4).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs)

        # Single camera: images is a direct tensor
        assert obs.images.shape == (1, 4, 100, 100)

    def test_already_torch_tensor(self):
        """Test handling when input is already a torch tensor.

        Note: Single camera returns direct tensor, not dict.
        """
        gym_obs = {
            "pixels": torch.rand(100, 100, 3),
            "agent_pos": torch.tensor([0.5, 0.3]),
        }

        obs = gym_observation_to_observation(gym_obs)

        # Single camera: images is a direct tensor
        assert obs.images.shape == (1, 3, 100, 100)
        assert isinstance(obs.state, torch.Tensor)
        assert obs.state.shape == (1, 2)

    def test_preserves_data_values(self):
        """Test that data values are preserved during conversion.

        Note: Single camera returns direct tensor, not dict.
        """
        pixels = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)  # 1x1x3
        state = np.array([0.7, 0.8], dtype=np.float32)

        gym_obs = {"pixels": pixels, "agent_pos": state}
        obs = gym_observation_to_observation(gym_obs)

        # Check state values
        assert isinstance(obs.state, torch.Tensor)
        assert obs.state[0, 0].item() == pytest.approx(0.7)
        assert obs.state[0, 1].item() == pytest.approx(0.8)

        # Check pixel values (after HWC->CHW conversion)
        # Single camera: images is a direct tensor
        assert obs.images[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert obs.images[0, 1, 0, 0].item() == pytest.approx(2.0)
        assert obs.images[0, 2, 0, 0].item() == pytest.approx(3.0)

    def test_multiple_camera_keys(self):
        """Test with multiple camera keys.

        When len(camera_keys) > 1, returns dict with first key.
        Only returns direct tensor when len(camera_keys) == 1.
        """
        gym_obs = {"pixels": np.random.rand(50, 50, 3).astype(np.float32)}

        # Multiple camera keys specified: returns dict with first key
        obs = gym_observation_to_observation(gym_obs, camera_keys=["cam1", "cam2", "cam3"])

        # Multiple camera keys: images is a dict with first key
        assert isinstance(obs.images, dict)
        assert "cam1" in obs.images
        assert obs.images["cam1"].shape == (1, 3, 50, 50)

    def test_none_camera_keys_uses_default(self):
        """Test that None camera_keys uses 'top' as default.

        Note: Single camera returns direct tensor, not dict.
        """
        gym_obs = {"pixels": np.random.rand(50, 50, 3).astype(np.float32)}

        obs = gym_observation_to_observation(gym_obs, camera_keys=None)

        # Single camera: images is a direct tensor
        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (1, 3, 50, 50)

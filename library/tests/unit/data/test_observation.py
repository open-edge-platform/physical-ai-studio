# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Observation dataclass."""

import numpy as np
import pytest
import torch

from getiaction.data import Observation
from getiaction.data.lerobot import FormatConverter


class TestObservationCreation:
    """Test Observation instantiation and basic properties."""

    def test_create_minimal_observation(self):
        """Test creating observation with minimal required fields."""
        obs = Observation()
        assert obs.action is None
        assert obs.state is None
        assert obs.images is None

    def test_create_with_tensors(self):
        """Test creating observation with tensor fields."""
        action = torch.tensor([1.0, 2.0])
        state = torch.tensor([0.5, 0.6])

        obs = Observation(action=action, state=state)

        assert torch.equal(obs.action, action)
        assert torch.equal(obs.state, state)

    def test_create_with_numpy(self):
        """Test creating observation with numpy arrays."""
        action = np.array([1.0, 2.0])
        state = np.array([0.5, 0.6])

        obs = Observation(action=action, state=state)

        assert np.array_equal(obs.action, action)
        assert np.array_equal(obs.state, state)

    def test_create_with_images_dict(self):
        """Test creating observation with multi-camera images."""
        images = {
            "top": torch.rand(3, 224, 224),
            "wrist": torch.rand(3, 224, 224),
        }

        obs = Observation(images=images)

        assert "top" in obs.images
        assert "wrist" in obs.images
        assert obs.images["top"].shape == (3, 224, 224)

    def test_create_with_metadata(self):
        """Test creating observation with metadata fields."""
        obs = Observation(
            episode_index=torch.tensor(5),
            frame_index=torch.tensor(10),
            index=torch.tensor(100),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(1.5),
        )

        assert obs.episode_index.item() == 5
        assert obs.frame_index.item() == 10
        assert obs.index.item() == 100

    def test_frozen_dataclass(self):
        """Test that Observation is immutable."""
        obs = Observation(action=torch.tensor([1.0, 2.0]))

        with pytest.raises(AttributeError):
            obs.action = torch.tensor([3.0, 4.0])


class TestObservationToDict:
    """Test Observation.to_dict() method."""

    def test_to_dict_basic(self):
        """Test converting observation to dictionary."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5]),
        )

        obs_dict = obs.to_dict()

        assert isinstance(obs_dict, dict)
        assert "action" in obs_dict
        assert "state" in obs_dict
        assert torch.equal(obs_dict["action"], obs.action)

    def test_to_dict_with_nested_images(self):
        """Test to_dict preserves nested structure."""
        images = {
            "top": torch.rand(3, 224, 224),
            "wrist": torch.rand(3, 224, 224),
        }
        obs = Observation(images=images)

        obs_dict = obs.to_dict(flatten=False)

        assert isinstance(obs_dict["images"], dict)
        assert "top" in obs_dict["images"]
        assert "wrist" in obs_dict["images"]

    def test_to_flat_dict_with_nested_images(self):
        """Test to_dict preserves nested structure."""
        images = {
            "top": torch.rand(3, 224, 224),
            "wrist": torch.rand(3, 224, 224),
        }
        obs = Observation(images=images)

        obs_dict = obs.to_dict(flatten=True)

        assert "images" not in obs_dict
        assert "images.top" in obs_dict
        assert "images.wrist" in obs_dict

        for k in Observation.get_all_component_dict_keys(obs_dict, Observation.ComponentKeys.IMAGES):
            assert k in obs_dict

    def test_to_dict_includes_none_fields(self):
        """Test to_dict includes None fields."""
        obs = Observation(action=torch.tensor([1.0]))
        obs_dict = obs.to_dict()

        assert "state" in obs_dict
        assert obs_dict["state"] is None
        assert "images" in obs_dict
        assert obs_dict["images"] is None


class TestObservationFromDict:
    """Test Observation.from_dict() class method."""

    def test_from_dict_basic(self):
        """Test creating observation from dictionary."""
        data = {
            "action": torch.tensor([1.0, 2.0]),
            "state": torch.tensor([0.5]),
        }

        obs = Observation.from_dict(data)

        assert torch.equal(obs.action, data["action"])
        assert torch.equal(obs.state, data["state"])

    def test_from_dict_filters_unknown_fields(self):
        """Test from_dict filters out unknown fields."""
        data = {
            "action": torch.tensor([1.0, 2.0]),
            "unknown_field": "this should be ignored",
            "another_unknown": 123,
        }

        obs = Observation.from_dict(data)

        assert obs.action is not None
        assert not hasattr(obs, "unknown_field")

    def test_from_dict_with_nested_images(self):
        """Test from_dict handles nested images."""
        data = {
            "images": {
                "top": torch.rand(3, 224, 224),
                "wrist": torch.rand(3, 224, 224),
            },
            "action": torch.tensor([1.0, 2.0]),
        }

        obs = Observation.from_dict(data)

        assert isinstance(obs.images, dict)
        assert "top" in obs.images
        assert "wrist" in obs.images

    def test_roundtrip_to_dict_from_dict(self):
        """Test to_dict → from_dict roundtrip."""
        original = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
            images={"top": torch.rand(3, 64, 64)},
            episode_index=torch.tensor(5),
        )

        obs_dict = original.to_dict()
        restored = Observation.from_dict(obs_dict)

        assert torch.equal(restored.action, original.action)
        assert torch.equal(restored.state, original.state)
        assert torch.equal(restored.episode_index, original.episode_index)


class TestObservationBatching:
    """Test Observation works for both single and batched data."""

    def test_single_observation_shapes(self):
        """Test shapes for unbatched observation."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),  # [action_dim]
            state=torch.tensor([0.5, 0.6]),    # [state_dim]
            images={"top": torch.rand(3, 64, 64)},  # [C, H, W]
        )

        assert obs.action.shape == (2,)
        assert obs.state.shape == (2,)
        assert obs.images["top"].shape == (3, 64, 64)

    def test_batched_observation_shapes(self):
        """Test shapes for batched observation."""
        obs = Observation(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # [B, action_dim]
            state=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),   # [B, state_dim]
            images={"top": torch.rand(2, 3, 64, 64)},  # [B, C, H, W]
        )

        assert obs.action.shape == (2, 2)  # batch_size=2
        assert obs.state.shape == (2, 2)
        assert obs.images["top"].shape == (2, 3, 64, 64)

    def test_observation_type_consistency(self):
        """Test same type used for single and batch."""
        single = Observation(action=torch.tensor([1.0, 2.0]))
        batch = Observation(action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        assert type(single) == type(batch)
        assert isinstance(single, Observation)
        assert isinstance(batch, Observation)


class TestObservationFormatConversion:
    """Test Observation works with FormatConverter."""

    def test_format_converter_to_lerobot_dict(self):
        """Test FormatConverter.to_lerobot_dict with Observation."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
            images={"top": torch.rand(3, 64, 64)},
            episode_index=torch.tensor(5),
            frame_index=torch.tensor(10),
            index=torch.tensor(100),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(1.5),
        )

        lerobot_dict = FormatConverter.to_lerobot_dict(obs)

        assert isinstance(lerobot_dict, dict)
        assert "action" in lerobot_dict
        assert "observation.state" in lerobot_dict
        assert "observation.images.top" in lerobot_dict
        assert torch.equal(lerobot_dict["action"], obs.action)

    def test_format_converter_to_observation(self):
        """Test FormatConverter.to_observation with dict."""
        lerobot_dict = {
            "action": torch.tensor([1.0, 2.0]),
            "observation.state": torch.tensor([0.5, 0.6]),
            "observation.images.top": torch.rand(3, 64, 64),
            "episode_index": torch.tensor(5),
            "frame_index": torch.tensor(10),
            "index": torch.tensor(100),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(1.5),
        }

        obs = FormatConverter.to_observation(lerobot_dict)

        assert isinstance(obs, Observation)
        assert torch.equal(obs.action, lerobot_dict["action"])
        assert torch.equal(obs.state, lerobot_dict["observation.state"])

    def test_format_converter_roundtrip(self):
        """Test Observation → LeRobot dict → Observation roundtrip."""
        original = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
            images={"top": torch.rand(3, 64, 64)},
            episode_index=torch.tensor(5),
            frame_index=torch.tensor(10),
            index=torch.tensor(100),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(1.5),
        )

        lerobot_dict = FormatConverter.to_lerobot_dict(original)
        restored = FormatConverter.to_observation(lerobot_dict)

        assert isinstance(restored, Observation)
        assert torch.equal(restored.action, original.action)
        assert torch.equal(restored.state, original.state)

    def test_format_converter_with_batched_observation(self):
        """Test FormatConverter works with batched Observation."""
        batch = Observation(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            state=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
            images={"top": torch.rand(2, 3, 64, 64)},
            episode_index=torch.tensor([5, 6]),
            frame_index=torch.tensor([10, 11]),
            index=torch.tensor([100, 101]),
            task_index=torch.tensor([0, 0]),
            timestamp=torch.tensor([1.5, 1.6]),
        )

        lerobot_dict = FormatConverter.to_lerobot_dict(batch)

        assert lerobot_dict["action"].shape == (2, 2)
        assert lerobot_dict["observation.state"].shape == (2, 2)
        assert lerobot_dict["observation.images.top"].shape == (2, 3, 64, 64)


class TestObservationEdgeCases:
    """Test edge cases and special scenarios."""

    def test_observation_with_single_image_tensor(self):
        """Test observation with single image tensor (not dict)."""
        obs = Observation(images=torch.rand(3, 64, 64))

        assert isinstance(obs.images, torch.Tensor)
        assert obs.images.shape == (3, 64, 64)

    def test_observation_with_dict_action(self):
        """Test observation with action as dict."""
        action_dict = {
            "gripper": torch.tensor([1.0]),
            "arm": torch.tensor([0.5, 0.6, 0.7]),
        }
        obs = Observation(action=action_dict)

        assert isinstance(obs.action, dict)
        assert "gripper" in obs.action
        assert "arm" in obs.action

    def test_observation_with_extra_fields(self):
        """Test observation with extra metadata."""
        extra = {"custom_field": "value", "another": 123}
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            extra=extra,
        )

        assert obs.extra == extra
        assert obs.extra["custom_field"] == "value"

    def test_observation_with_info_field(self):
        """Test observation with info metadata."""
        info = {"episode_id": "abc123", "trial": 5}
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            info=info,
        )

        assert obs.info == info
        assert obs.info["episode_id"] == "abc123"

    def test_empty_observation(self):
        """Test creating completely empty observation."""
        obs = Observation()

        assert obs.action is None
        assert obs.state is None
        assert obs.images is None
        assert obs.episode_index is None


class TestObservationAttributeAccess:
    """Test attribute access patterns."""

    def test_direct_attribute_access(self):
        """Test accessing fields as attributes."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5]),
        )

        # Can access as attributes
        action = obs.action
        state = obs.state

        assert torch.equal(action, torch.tensor([1.0, 2.0]))
        assert torch.equal(state, torch.tensor([0.5]))

    def test_nested_dict_access(self):
        """Test accessing nested dictionaries."""
        images = {
            "top": torch.rand(3, 64, 64),
            "wrist": torch.rand(3, 64, 64),
        }
        obs = Observation(images=images)

        # Can access nested dicts
        top_image = obs.images["top"]
        wrist_image = obs.images["wrist"]

        assert top_image.shape == (3, 64, 64)
        assert wrist_image.shape == (3, 64, 64)

    def test_hasattr_checks(self):
        """Test hasattr works on Observation."""
        obs = Observation(action=torch.tensor([1.0]))

        assert hasattr(obs, "action")
        assert hasattr(obs, "state")
        assert hasattr(obs, "images")
        assert not hasattr(obs, "nonexistent_field")


class TestObservationWithOptionalFields:
    """Test optional RL and metadata fields."""

    def test_observation_with_reward(self):
        """Test observation with next_reward field."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            next_reward=torch.tensor(0.5),
        )

        assert obs.next_reward is not None
        assert obs.next_reward.item() == 0.5

    def test_observation_with_success(self):
        """Test observation with next_success field."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            next_success=True,
        )

        assert obs.next_success is True

    def test_observation_with_task(self):
        """Test observation with task field."""
        task = torch.tensor([1, 0, 0])  # One-hot encoded task
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            task=task,
        )

        assert torch.equal(obs.task, task)

    def test_observation_all_metadata_fields(self):
        """Test observation with all metadata fields."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            episode_index=torch.tensor(5),
            frame_index=torch.tensor(10),
            index=torch.tensor(100),
            task_index=torch.tensor(0),
            timestamp=torch.tensor(1.5),
            next_reward=torch.tensor(0.5),
            next_success=True,
            info={"key": "value"},
            extra={"custom": 123},
        )

        assert obs.episode_index.item() == 5
        assert obs.frame_index.item() == 10
        assert obs.next_reward.item() == 0.5
        assert obs.next_success is True
        assert obs.info["key"] == "value"
        assert obs.extra["custom"] == 123


class TestObservationDeviceTransfer:
    """Test Observation.to() method for device transfer."""

    def test_to_device_single_tensor(self):
        """Test moving observation with single tensor to device."""
        obs = Observation(action=torch.tensor([1.0, 2.0]))
        obs_moved = obs.to("cpu")

        assert obs_moved.action.device.type == "cpu"
        # Original unchanged (immutable)
        assert obs is not obs_moved

    def test_to_device_multiple_tensors(self):
        """Test moving observation with multiple tensors."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
        )
        obs_moved = obs.to("cpu")

        assert obs_moved.action.device.type == "cpu"
        assert obs_moved.state.device.type == "cpu"

    def test_to_device_nested_dict(self):
        """Test moving observation with nested dict (multi-camera)."""
        obs = Observation(
            action=torch.tensor([1.0]),
            images={"top": torch.rand(3, 64, 64), "wrist": torch.rand(3, 32, 32)},
        )
        obs_moved = obs.to("cpu")

        assert obs_moved.images["top"].device.type == "cpu"
        assert obs_moved.images["wrist"].device.type == "cpu"

    def test_to_device_preserves_none(self):
        """Test that None fields remain None after device transfer."""
        obs = Observation(action=torch.tensor([1.0]), state=None, images=None)
        obs_moved = obs.to("cpu")

        assert obs_moved.action.device.type == "cpu"
        assert obs_moved.state is None
        assert obs_moved.images is None

    def test_to_device_preserves_shapes(self):
        """Test that tensor shapes are preserved during device transfer."""
        action_shape = (8, 2)
        state_shape = (8, 10)
        images_shape = (8, 3, 224, 224)

        obs = Observation(
            action=torch.randn(action_shape),
            state=torch.randn(state_shape),
            images={"top": torch.rand(images_shape)},
        )
        obs_moved = obs.to("cpu")

        assert obs_moved.action.shape == action_shape
        assert obs_moved.state.shape == state_shape
        assert obs_moved.images["top"].shape == images_shape

    def test_to_device_preserves_non_tensor_fields(self):
        """Test that non-tensor fields are preserved."""
        obs = Observation(
            action=torch.tensor([1.0]),
            next_success=True,
            info={"key": "value"},
        )
        obs_moved = obs.to("cpu")

        assert obs_moved.next_success is True
        assert obs_moved.info["key"] == "value"

    def test_to_device_immutability(self):
        """Test that original observation is not modified."""
        obs = Observation(action=torch.tensor([1.0, 2.0]))
        original_device = obs.action.device

        obs_moved = obs.to("cpu")

        # Original unchanged
        assert obs.action.device == original_device
        # New instance created
        assert obs is not obs_moved


class TestGymInValidation:
    """Tests for Gym usage in validation (no wrapper needed)."""

    def test_gym_direct_usage(self):
        """Test that Gym can be used directly in validation."""
        from getiaction.gyms import Gym, PushTGym

        gym = PushTGym()

        assert isinstance(gym, Gym)
        assert hasattr(gym, "reset")
        assert hasattr(gym, "step")

    def test_gym_reset_returns_observation(self):
        """Test that Gym.reset() returns observation dict."""
        from getiaction.gyms import PushTGym

        gym = PushTGym()
        observation, info = gym.reset(seed=42)

        assert isinstance(observation, dict)
        assert "pixels" in observation or "state" in observation

    def test_gym_step_returns_observation(self):
        """Test that Gym.step() returns observation dict."""
        import numpy as np

        from getiaction.gyms import PushTGym

        gym = PushTGym()
        gym.reset(seed=42)

        action_shape = gym.action_space.shape
        assert action_shape is not None
        action = np.zeros(action_shape)
        observation, reward, terminated, truncated, info = gym.step(action)

        assert isinstance(observation, dict)
        assert isinstance(reward, (int, float, np.number))

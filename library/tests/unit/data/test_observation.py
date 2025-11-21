# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Observation dataclass."""

import numpy as np
import pytest
import torch

from getiaction.data.lerobot import FormatConverter
from getiaction.data.observation import IMAGES, Observation


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

        for k in Observation.get_flattened_keys(obs_dict, field=IMAGES):
            assert k in obs_dict

    def test_get_flattened_keys_fallback(self):
        data = {
            "images.top": torch.rand(3, 224, 224),
            "images.wrist": torch.rand(3, 224, 224),
        }

        keys = Observation.get_flattened_keys(data, field="images")

        assert "images.top" in keys
        assert "images.wrist" in keys

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


class TestObservationNumpyTensorConversion:
    """Test Observation.to_numpy() and to_torch() conversion methods."""

    def test_to_numpy_single_tensor(self):
        """Test converting single torch tensor to numpy."""
        obs = Observation(action=torch.tensor([1.0, 2.0]))
        obs_np = obs.to_numpy()

        assert isinstance(obs_np.action, np.ndarray)
        assert np.array_equal(obs_np.action, np.array([1.0, 2.0]))

    def test_to_numpy_multiple_tensors(self):
        """Test converting multiple torch tensors to numpy."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
        )
        obs_np = obs.to_numpy()

        assert isinstance(obs_np.action, np.ndarray)
        assert isinstance(obs_np.state, np.ndarray)
        assert np.allclose(obs_np.action, np.array([1.0, 2.0]))
        assert np.allclose(obs_np.state, np.array([0.5, 0.6]))

    def test_to_numpy_nested_dict(self):
        """Test converting nested dict of tensors to numpy."""
        obs = Observation(
            action=torch.tensor([1.0]),
            images={"top": torch.rand(3, 64, 64), "wrist": torch.rand(3, 32, 32)},
        )
        obs_np = obs.to_numpy()

        assert isinstance(obs_np.images["top"], np.ndarray)
        assert isinstance(obs_np.images["wrist"], np.ndarray)
        assert obs_np.images["top"].shape == (3, 64, 64)
        assert obs_np.images["wrist"].shape == (3, 32, 32)

    def test_to_numpy_preserves_existing_numpy(self):
        """Test that existing numpy arrays are preserved."""
        obs = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=np.array([0.5, 0.6]),
        )
        obs_np = obs.to_numpy()

        assert isinstance(obs_np.action, np.ndarray)
        assert isinstance(obs_np.state, np.ndarray)
        # Both should be numpy now
        assert np.array_equal(obs_np.state, np.array([0.5, 0.6]))

    def test_to_numpy_preserves_none(self):
        """Test that None fields remain None."""
        obs = Observation(action=torch.tensor([1.0]), state=None, images=None)
        obs_np = obs.to_numpy()

        assert isinstance(obs_np.action, np.ndarray)
        assert obs_np.state is None
        assert obs_np.images is None

    def test_to_numpy_preserves_shapes(self):
        """Test that shapes are preserved during conversion."""
        action_shape = (8, 2)
        state_shape = (8, 10)

        obs = Observation(
            action=torch.randn(action_shape),
            state=torch.randn(state_shape),
        )
        obs_np = obs.to_numpy()

        assert obs_np.action.shape == action_shape
        assert obs_np.state.shape == state_shape

    def test_to_numpy_immutability(self):
        """Test that original observation is not modified."""
        obs = Observation(action=torch.tensor([1.0, 2.0]))
        obs_np = obs.to_numpy()

        assert isinstance(obs.action, torch.Tensor)
        assert isinstance(obs_np.action, np.ndarray)
        assert obs is not obs_np

    def test_to_torch_single_array(self):
        """Test converting single numpy array to torch."""
        obs = Observation(action=np.array([1.0, 2.0]))
        obs_torch = obs.to_torch()

        assert isinstance(obs_torch.action, torch.Tensor)
        assert torch.equal(obs_torch.action, torch.tensor([1.0, 2.0]))

    def test_to_torch_multiple_arrays(self):
        """Test converting multiple numpy arrays to torch."""
        obs = Observation(
            action=np.array([1.0, 2.0]),
            state=np.array([0.5, 0.6]),
        )
        obs_torch = obs.to_torch()

        assert isinstance(obs_torch.action, torch.Tensor)
        assert isinstance(obs_torch.state, torch.Tensor)
        assert torch.allclose(obs_torch.action, torch.tensor([1.0, 2.0], dtype=obs_torch.action.dtype))
        assert torch.allclose(obs_torch.state, torch.tensor([0.5, 0.6], dtype=obs_torch.state.dtype))

    def test_to_torch_nested_dict(self):
        """Test converting nested dict of arrays to torch."""
        obs = Observation(
            action=np.array([1.0]),
            images={"top": np.random.rand(3, 64, 64), "wrist": np.random.rand(3, 32, 32)},
        )
        obs_torch = obs.to_torch()

        assert isinstance(obs_torch.images["top"], torch.Tensor)
        assert isinstance(obs_torch.images["wrist"], torch.Tensor)
        assert obs_torch.images["top"].shape == (3, 64, 64)
        assert obs_torch.images["wrist"].shape == (3, 32, 32)

    def test_to_torch_preserves_existing_torch(self):
        """Test that existing torch tensors are preserved."""
        obs = Observation(
            action=np.array([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
        )
        obs_torch = obs.to_torch()

        assert isinstance(obs_torch.action, torch.Tensor)
        assert isinstance(obs_torch.state, torch.Tensor)
        # Both should be torch now
        assert torch.equal(obs_torch.state, torch.tensor([0.5, 0.6]))

    def test_to_torch_preserves_none(self):
        """Test that None fields remain None."""
        obs = Observation(action=np.array([1.0]), state=None, images=None)
        obs_torch = obs.to_torch()

        assert isinstance(obs_torch.action, torch.Tensor)
        assert obs_torch.state is None
        assert obs_torch.images is None

    def test_to_torch_preserves_shapes(self):
        """Test that shapes are preserved during conversion."""
        action_shape = (8, 2)
        state_shape = (8, 10)

        obs = Observation(
            action=np.random.randn(*action_shape),
            state=np.random.randn(*state_shape),
        )
        obs_torch = obs.to_torch()

        assert obs_torch.action.shape == action_shape
        assert obs_torch.state.shape == state_shape

    def test_to_torch_immutability(self):
        """Test that original observation is not modified."""
        obs = Observation(action=np.array([1.0, 2.0]))
        obs_torch = obs.to_torch()

        assert isinstance(obs.action, np.ndarray)
        assert isinstance(obs_torch.action, torch.Tensor)
        assert obs is not obs_torch

    def test_roundtrip_torch_numpy_torch(self):
        """Test torch → numpy → torch roundtrip."""
        original = Observation(
            action=torch.tensor([1.0, 2.0]),
            state=torch.tensor([0.5, 0.6]),
            images={"top": torch.rand(3, 64, 64)},
        )

        # Convert to numpy and back
        obs_np = original.to_numpy()
        obs_back = obs_np.to_torch()

        # Verify types
        assert isinstance(obs_np.action, np.ndarray)
        assert isinstance(obs_back.action, torch.Tensor)

        # Verify values (approximately equal for floating point)
        assert torch.allclose(obs_back.action, original.action)
        assert torch.allclose(obs_back.state, original.state)
        assert torch.allclose(obs_back.images["top"], original.images["top"])

    def test_roundtrip_numpy_torch_numpy(self):
        """Test numpy → torch → numpy roundtrip."""
        original = Observation(
            action=np.array([1.0, 2.0]),
            state=np.array([0.5, 0.6]),
            images={"top": np.random.rand(3, 64, 64)},
        )

        # Convert to torch and back
        obs_torch = original.to_torch()
        obs_back = obs_torch.to_numpy()

        # Verify types
        assert isinstance(obs_torch.action, torch.Tensor)
        assert isinstance(obs_back.action, np.ndarray)

        # Verify values
        assert np.allclose(obs_back.action, original.action)
        assert np.allclose(obs_back.state, original.state)
        assert np.allclose(obs_back.images["top"], original.images["top"])

    def test_to_numpy_with_batched_observation(self):
        """Test to_numpy with batched observations."""
        batch_size = 8
        obs = Observation(
            action=torch.randn(batch_size, 2),
            state=torch.randn(batch_size, 10),
            images={"top": torch.rand(batch_size, 3, 224, 224)},
        )
        obs_np = obs.to_numpy()

        assert obs_np.action.shape == (batch_size, 2)
        assert obs_np.state.shape == (batch_size, 10)
        assert obs_np.images["top"].shape == (batch_size, 3, 224, 224)

    def test_to_torch_with_batched_observation(self):
        """Test to_torch with batched observations."""
        batch_size = 8
        obs = Observation(
            action=np.random.randn(batch_size, 2),
            state=np.random.randn(batch_size, 10),
            images={"top": np.random.rand(batch_size, 3, 224, 224)},
        )
        obs_torch = obs.to_torch()

        assert obs_torch.action.shape == (batch_size, 2)
        assert obs_torch.state.shape == (batch_size, 10)
        assert obs_torch.images["top"].shape == (batch_size, 3, 224, 224)

    def test_conversion_preserves_non_tensor_fields(self):
        """Test that non-tensor/array fields are preserved during conversion."""
        obs = Observation(
            action=torch.tensor([1.0]),
            next_success=True,
            info={"key": "value"},
        )

        # To numpy
        obs_np = obs.to_numpy()
        assert obs_np.next_success is True
        assert obs_np.info["key"] == "value"

        # To torch
        obs_torch = obs_np.to_torch()
        assert obs_torch.next_success is True
        assert obs_torch.info["key"] == "value"


class TestObservationIndexing:
    """Test Observation.__getitem__() method for indexing batched observations."""

    def test_integer_indexing(self):
        """Test integer indexing extracts single sample."""
        batch = Observation(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            state=torch.tensor([[0.1], [0.2], [0.3]]),
            images={"top": torch.rand(3, 3, 64, 64)},
        )

        sample = batch[1]

        assert isinstance(sample, Observation)
        assert sample.action.shape == (2,)
        assert torch.equal(sample.action, torch.tensor([3.0, 4.0]))
        assert sample.state.shape == (1,)
        assert sample.images["top"].shape == (3, 64, 64)

    def test_negative_indexing(self):
        """Test negative indexing works correctly."""
        batch = Observation(action=torch.tensor([[1.0], [2.0], [3.0]]))

        assert torch.equal(batch[-1].action, torch.tensor([3.0]))
        assert torch.equal(batch[-2].action, torch.tensor([2.0]))

    def test_slice_indexing(self):
        """Test slice indexing extracts sub-batch."""
        batch = Observation(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            state=torch.randn(4, 5),
        )

        sub_batch = batch[1:3]

        assert sub_batch.action.shape == (2, 2)
        assert torch.equal(sub_batch.action, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        assert sub_batch.state.shape == (2, 5)

    def test_slice_preserves_batch_dimension(self):
        """Test slice can preserve batch dimension."""
        batch = Observation(action=torch.randn(8, 2))

        sample = batch[0:1]

        assert sample.action.shape == (1, 2)
        assert sample.action.ndim == 2

    def test_slice_with_step(self):
        """Test slice indexing with step parameter."""
        batch = Observation(action=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]))

        sub_batch = batch[::2]

        assert torch.equal(sub_batch.action, torch.tensor([[1.0], [3.0], [5.0]]))

    def test_nested_dict_indexing(self):
        """Test indexing with nested dictionaries."""
        batch = Observation(
            action={"gripper": torch.randn(4, 1), "arm": torch.randn(4, 6)},
            images={"cam1": torch.rand(4, 3, 64, 64), "cam2": torch.rand(4, 3, 32, 32)},
        )

        sample = batch[2]

        assert sample.action["gripper"].shape == (1,)
        assert sample.action["arm"].shape == (6,)
        assert sample.images["cam1"].shape == (3, 64, 64)
        assert sample.images["cam2"].shape == (3, 32, 32)

    def test_preserves_none_fields(self):
        """Test None fields remain None after indexing."""
        batch = Observation(action=torch.randn(5, 2), state=None, images=None)

        sample = batch[0]

        assert sample.state is None
        assert sample.images is None

    def test_preserves_non_indexable_fields(self):
        """Test non-indexable fields pass through unchanged."""
        batch = Observation(
            action=torch.randn(3, 2),
            next_success=True,
            info={"key": "value"},
            extra={"custom": 123},
        )

        sample = batch[0]

        assert sample.next_success is True
        assert sample.info["key"] == "value"
        assert sample.extra["custom"] == 123

    def test_numpy_array_indexing(self):
        """Test indexing works with numpy arrays."""
        batch = Observation(
            action=np.array([[1.0, 2.0], [3.0, 4.0]]),
            state=np.array([[0.1], [0.2]]),
        )

        sample = batch[1]

        assert isinstance(sample.action, np.ndarray)
        assert np.array_equal(sample.action, np.array([3.0, 4.0]))
        assert np.array_equal(sample.state, np.array([0.2]))

    def test_mixed_tensor_array_types(self):
        """Test indexing with mixed torch and numpy types."""
        batch = Observation(
            action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            state=np.array([[0.1], [0.2]]),
        )

        sample = batch[0]

        assert isinstance(sample.action, torch.Tensor)
        assert isinstance(sample.state, np.ndarray)

    def test_indexing_immutability(self):
        """Test indexing creates new instance without modifying original."""
        batch = Observation(action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        original_shape = batch.action.shape

        sample = batch[0]

        assert batch.action.shape == original_shape
        assert batch is not sample

    def test_metadata_fields_indexed(self):
        """Test metadata fields are properly indexed."""
        batch = Observation(
            action=torch.randn(3, 2),
            episode_index=torch.tensor([0, 0, 1]),
            frame_index=torch.tensor([10, 11, 12]),
            timestamp=torch.tensor([1.0, 1.1, 2.0]),
        )

        sample = batch[1]

        assert sample.episode_index.item() == 0
        assert sample.frame_index.item() == 11
        assert sample.timestamp.item() == 1.1

    def test_out_of_bounds_raises(self):
        """Test out of bounds index raises IndexError."""
        batch = Observation(action=torch.randn(3, 2))

        with pytest.raises(IndexError):
            _ = batch[10]

    def test_dataloader_usage_pattern(self):
        """Test typical DataLoader batch manipulation."""
        batch = Observation(
            action=torch.randn(16, 2),
            images={"top": torch.rand(16, 3, 224, 224)},
        )

        # Extract mini-batch
        mini_batch = batch[0:8]
        assert mini_batch.action.shape == (8, 2)

        # Extract single sample
        single = mini_batch[0]
        assert single.action.shape == (2,)


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

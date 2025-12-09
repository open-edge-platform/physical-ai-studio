# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for all tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def dummy_dataset():
    """Create a simple dummy dataset for testing.

    Returns a dataset that mimics the structure of a real dataset
    without requiring any external data files.
    """
    from getiaction.data import Dataset
    from getiaction.data.observation import (
        Feature,
        FeatureType,
        NormalizationParameters,
        Observation,
    )

    class DummyDataset(Dataset):
        """Simple in-memory dataset for testing.

        This dataset properly implements the getiaction.data.Dataset interface
        including all required properties (raw_features, fps, tolerance_s, delta_indices).
        """

        def __init__(
            self, num_samples: int = 10, state_dim: int = 4, action_dim: int = 2
        ):
            self.num_samples = num_samples
            self.state_dim = state_dim
            self.action_dim = action_dim
            self._delta_indices: dict[str, list[int]] = {}

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Check if action chunks are requested via delta_indices
            num_action_steps = len(self._delta_indices.get("action", [0]))

            # Return action chunks if delta_indices specifies multiple steps
            action_shape = (
                (num_action_steps, self.action_dim)
                if num_action_steps > 1
                else (self.action_dim,)
            )

            # Create extra dict with action_is_pad if using action chunks
            extra = None
            if num_action_steps > 1:
                # For dummy dataset, no padding - all actions are valid (False = not padded)
                extra = {
                    "action_is_pad": torch.full(
                        (num_action_steps,), fill_value=False, dtype=torch.bool
                    )
                }

            return Observation(
                action=torch.randn(*action_shape),
                state=torch.randn(self.state_dim),
                images=torch.randn(3, 96, 96),  # Direct tensor for single camera
                extra=extra,
            )

        @property
        def raw_features(self) -> dict:
            """Return raw dataset features (mimics HuggingFace format)."""
            return {
                "action": {"shape": (self.action_dim,), "dtype": "float32"},
                "observation.state": {"shape": (self.state_dim,), "dtype": "float32"},
                "observation.images.camera": {
                    "shape": (96, 96, 3),  # HF format is (H, W, C)
                    "dtype": "video",
                    "names": ["height", "width", "channels"],
                },
            }

        @property
        def action_features(self) -> dict[str, Feature]:
            """Return action features for the dummy dataset."""
            return {
                "action": Feature(
                    ftype=FeatureType.ACTION,
                    shape=(self.action_dim,),
                    name="action",
                    normalization_data=NormalizationParameters(
                        mean=[0.0] * self.action_dim,
                        std=[1.0] * self.action_dim,
                        min=[-1.0] * self.action_dim,
                        max=[1.0] * self.action_dim,
                    ),
                ),
            }

        @property
        def observation_features(self) -> dict[str, Feature]:
            """Return observation features for the dummy dataset."""
            return {
                "state": Feature(
                    ftype=FeatureType.STATE,
                    shape=(self.state_dim,),
                    name="state",
                    normalization_data=NormalizationParameters(
                        mean=[0.0] * self.state_dim,
                        std=[1.0] * self.state_dim,
                        min=[-1.0] * self.state_dim,
                        max=[1.0] * self.state_dim,
                    ),
                ),
                "camera": Feature(
                    ftype=FeatureType.VISUAL,
                    shape=(3, 96, 96),
                    name="camera",
                    normalization_data=NormalizationParameters(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        min=[0.0, 0.0, 0.0],
                        max=[1.0, 1.0, 1.0],
                    ),
                ),
            }

        @property
        def fps(self) -> int:
            """Frames per second of the dataset."""
            return 10

        @property
        def tolerance_s(self) -> float:
            """Tolerance to keep delta timestamps in sync with fps."""
            return 1e-4

        @property
        def delta_indices(self) -> dict[str, list[int]]:
            """Expose delta_indices from the dataset."""
            return self._delta_indices

        @delta_indices.setter
        def delta_indices(self, indices: dict[str, list[int]]) -> None:
            """Allow setting delta_indices on the dataset."""
            self._delta_indices = indices

    return DummyDataset


@pytest.fixture
def dummy_lerobot_dataset():
    """Create a dummy LeRobot-compatible dataset for testing.

    This dataset mimics the structure of a LeRobotDataset but doesn't require
    downloading any data. It's useful for fast tests that don't need real data.

    Returns:
        A class (not instance) that can be instantiated with num_samples parameter.
    """
    import torch

    class DummyLeRobotDataset:
        """Dummy dataset that mimics LeRobot dataset structure."""

        def __init__(self, num_samples: int = 100):
            self.num_samples = num_samples
            # Create meta object with features and stats
            self._meta = self._create_meta()

        @staticmethod
        def _create_meta():
            """Create metadata that mimics LeRobotDataset.meta."""
            from types import SimpleNamespace

            # Mock features (HuggingFace datasets.Features format - must be dict of dicts!)
            # LeRobot expects specific conventions:
            # - Images: dtype must be "image" or "video", shape (H, W, C), names for dimensions
            # - State: "observation.state" key, dtype "float32" or similar
            # - Action: "action" key, dtype "float32" or similar
            features_dict = {
                "observation.state": {"shape": (4,), "dtype": "float32"},
                "observation.images.top": {
                    "shape": (
                        96,
                        96,
                        3,
                    ),  # (H, W, C) format - will be converted to (C, H, W) by dataset_to_policy_features
                    "dtype": "video",
                    "names": ["height", "width", "channels"],
                },
                "action": {"shape": (2,), "dtype": "float32"},
                "episode_index": {"shape": (), "dtype": "int64"},
                "frame_index": {"shape": (), "dtype": "int64"},
                "timestamp": {"shape": (), "dtype": "float32"},
                "next.done": {"shape": (), "dtype": "bool"},
            }

            # Mock stats (normalization statistics)
            # Include stats for all features that will be used by the policy
            stats = {
                "observation.state": {
                    "mean": torch.zeros(4),
                    "std": torch.ones(4),
                    "min": torch.full((4,), -1.0),
                    "max": torch.full((4,), 1.0),
                },
                "observation.images.top": {
                    "mean": torch.zeros(
                        3, 1, 1
                    ),  # (C, 1, 1) for channel-wise normalization
                    "std": torch.ones(3, 1, 1),
                    "min": torch.zeros(3, 1, 1),
                    "max": torch.full((3, 1, 1), 255.0),
                },
                "action": {
                    "mean": torch.zeros(2),
                    "std": torch.ones(2),
                    "min": torch.full((2,), -1.0),
                    "max": torch.full((2,), 1.0),
                },
            }

            # Create meta object
            meta = SimpleNamespace(
                robot_type="dummy_robot",
                fps=10,
                encoding={"observation.images.top": {"type": "video"}},
                features=features_dict,  # Must be a dict!
                stats=stats,
            )

            return meta

        @property
        def meta(self):
            """Return metadata matching LeRobotDataset.meta structure."""
            return self._meta

        @property
        def features(self):
            """Return features for compatibility."""
            return self._meta.features

        @property
        def episode_data_index(self):
            """Mimic episode data index structure."""
            # Simple mock: 10 frames per episode
            return {
                "from": [i * 10 for i in range(self.num_samples // 10)],
                "to": [(i + 1) * 10 for i in range(self.num_samples // 10)],
            }

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            """Return a sample mimicking LeRobot format."""
            return {
                "observation.state": torch.randn(4),
                "observation.images.top": torch.randn(3, 96, 96),
                "action": torch.randn(2),
                "episode_index": torch.tensor(idx // 10),
                "frame_index": torch.tensor(idx % 10),
                "timestamp": torch.tensor(idx * 0.1),
                "next.done": torch.tensor(False),
            }

    return DummyLeRobotDataset


@pytest.fixture
def dummy_datamodule(dummy_dataset):
    """Create a DataModule with dummy datasets for testing.

    Args:
        dummy_dataset: Fixture providing the DummyDataset class.

    Returns:
        Configured DataModule with dummy data.
    """
    from getiaction.data import DataModule
    from getiaction.gyms import PushTGym

    gym = PushTGym()
    train_dataset = dummy_dataset(num_samples=20)

    datamodule = DataModule(
        train_dataset=train_dataset,
        train_batch_size=4,
        val_gym=gym,
        num_rollouts_val=2,
    )

    return datamodule


@pytest.fixture
def dummy_lerobot_datamodule(dummy_lerobot_dataset):
    """Create a DataModule with dummy LeRobot-style dataset for testing.

    Note: This uses the standard DataModule (not LeRobotDataModule) because
    LeRobotDataModule requires an actual LeRobotDataset instance. The dummy
    dataset mimics LeRobot structure for testing purposes without downloads.

    For real LeRobot datasets, use LeRobotDataModule directly.

    Args:
        dummy_lerobot_dataset: Fixture providing the DummyLeRobotDataset class.

    Returns:
        Configured DataModule with dummy LeRobot-style dataset.
    """
    from getiaction.data import DataModule
    from getiaction.gyms import PushTGym

    gym = PushTGym()
    train_dataset = dummy_lerobot_dataset(num_samples=100)

    datamodule = DataModule(
        train_dataset=train_dataset,
        train_batch_size=8,
        val_gym=gym,
        num_rollouts_val=2,
    )

    return datamodule


@pytest.fixture
def pusht_gym():
    """Create PushT gym environment for testing."""
    from getiaction.gyms import PushTGym

    return PushTGym()

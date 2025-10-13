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
    from getiaction.data.observation import Feature, FeatureType, NormalizationParameters, Observation

    class DummyDataset:
        """Simple in-memory dataset for testing."""

        def __init__(self, num_samples: int = 10):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return Observation(
                action=torch.randn(2),
                state=torch.randn(4),
                images={"camera": torch.randn(3, 96, 96)},
            )

        @property
        def action_features(self) -> dict[str, Feature]:
            """Return action features for the dummy dataset."""
            return {
                "action": Feature(
                    ftype=FeatureType.ACTION,
                    shape=(2,),
                    name="action",
                    normalization_data=NormalizationParameters(
                        mean=torch.zeros(2),
                        std=torch.ones(2),
                        min=torch.full((2,), -1.0),
                        max=torch.full((2,), 1.0),
                    ),
                ),
            }

        @property
        def observation_features(self) -> dict[str, Feature]:
            """Return observation features for the dummy dataset."""
            return {
                "state": Feature(
                    ftype=FeatureType.STATE,
                    shape=(4,),
                    name="state",
                    normalization_data=NormalizationParameters(
                        mean=torch.zeros(4),
                        std=torch.ones(4),
                        min=torch.full((4,), -1.0),
                        max=torch.full((4,), 1.0),
                    ),
                ),
                "camera": Feature(
                    ftype=FeatureType.VISUAL,
                    shape=(3, 96, 96),
                    name="camera",
                    normalization_data=NormalizationParameters(
                        mean=torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1),
                        std=torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1),
                        min=torch.zeros(3, 1, 1),
                        max=torch.ones(3, 1, 1),
                    ),
                ),
            }

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
                    "shape": (96, 96, 3),  # (H, W, C) format - will be converted to (C, H, W) by dataset_to_policy_features
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
                    "mean": torch.zeros(3, 1, 1),  # (C, 1, 1) for channel-wise normalization
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
        val_gyms=gym,
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
        val_gyms=gym,
        num_rollouts_val=2,
    )

    return datamodule

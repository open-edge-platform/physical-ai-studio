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
    from getiaction.data.observation import Observation

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

    return DummyDataset


@pytest.fixture
def dummy_lerobot_dataset():
    """Create a dummy dataset that mimics LeRobot dataset structure.

    Returns a dataset compatible with LeRobot policies without requiring
    downloading actual LeRobot datasets.
    """

    class DummyLeRobotDataset:
        """Mimics LeRobot dataset structure for testing."""

        def __init__(self, num_samples: int = 100):
            self.num_samples = num_samples
            # Mimic LeRobot's dataset structure
            self.episode_data_index = {
                "from": torch.arange(0, num_samples, 10),
                "to": torch.arange(10, num_samples + 10, 10),
            }
            self.hf_dataset = None  # LeRobot uses HuggingFace datasets
            self.delta_timestamps = {
                "action": [0],
                "observation.state": [-1, 0],
                "observation.image": [0],
            }

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            """Return a batch mimicking LeRobot format."""
            return {
                "observation.state": torch.randn(1, 4),
                "observation.image": torch.randn(1, 3, 96, 96),
                "action": torch.randn(1, 2),
                "episode_index": torch.tensor([idx // 10]),
                "frame_index": torch.tensor([idx % 10]),
                "timestamp": torch.tensor([idx * 0.1]),
                "next.done": torch.tensor([False]),
            }

        @property
        def features(self):
            """Mimic HuggingFace datasets features."""
            return {
                "observation.state": {"shape": (4,), "dtype": "float32"},
                "observation.image": {"shape": (3, 96, 96), "dtype": "uint8"},
                "action": {"shape": (2,), "dtype": "float32"},
            }

        @property
        def meta(self):
            """Mimic LeRobot dataset metadata."""
            return {
                "robot_type": "dummy_robot",
                "fps": 10,
                "encoding": {"observation.image": {"type": "video"}},
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

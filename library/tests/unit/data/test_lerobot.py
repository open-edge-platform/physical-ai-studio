# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for lerobot dataset using a mock to avoid ffmpeg/network dependencies."""

from getiaction.data import Dataset, Observation
from getiaction.data.lerobot.dataset import _LeRobotDatasetAdapter
import torch
import pytest


class FakeLeRobotDataset:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.images.wrist": torch.randn(3, 64, 64),
            "observation.state": torch.randn(8),
            "action": torch.randn(7),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task.instructions": "pusht",
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
            "random": ["thing"]
        }


class FakeLeRobotDataset2:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.images.wrist": torch.randn(3, 64, 64),
            "observation.state": torch.randn(8),
            "action.continuous": torch.randn(7),
            "action.discrete": torch.randint(low=0, high=10, size=(7,)),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task.instructions": "pusht",
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }


class FakeLeRobotDataset_no_task_or_image:
    """A mock that mimics LeRobotDataset without needing ffmpeg or network access."""
    def __init__(self, repo_id=None, episodes=None, **kwargs):
        """Accepts arguments but does nothing with them."""
        self._length = 150  # A fixed length for our mock dataset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Returns a fake data dictionary, similar to the real dataset."""
        if idx >= self._length:
            raise IndexError("Index out of range")
        torch.manual_seed(idx)
        return {
            "observation.state": torch.randn(8),
            "action": torch.randint(low=0, high=10, size=(7,)),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }


@pytest.mark.parametrize(
    "dataset_cls",
    [FakeLeRobotDataset, FakeLeRobotDataset2, FakeLeRobotDataset_no_task_or_image],
)
class TestLeRobotActionDataset:
    """Groups tests for the LeRobotActionDataset wrapper, using multiple mock datasets."""

    @pytest.fixture
    def raw_lerobot_dataset(self, dataset_cls):
        """Fixture to provide a mock dataset instance for the current parameter."""
        return dataset_cls()

    def test_initialization(self, monkeypatch, dataset_cls):
        """Tests that LeRobotActionDataset initializes correctly by patching."""
        monkeypatch.setattr(
            "getiaction.data.lerobot.dataset.LeRobotDataset", dataset_cls
        )

        dataset = _LeRobotDatasetAdapter(repo_id="any/repo", episodes=[0])

        assert isinstance(dataset, Dataset)
        assert isinstance(dataset._lerobot_dataset, dataset_cls)
        assert len(dataset) > 0

    def test_len_delegation(self, raw_lerobot_dataset):
        """Tests that __len__ correctly delegates to the mock dataset."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)
        assert len(action_dataset) == len(raw_lerobot_dataset)
        assert len(action_dataset) == 150

    def test_getitem_returns_observation(self, raw_lerobot_dataset):
        """Tests that __getitem__ returns a correctly formatted Observation object."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)
        observation = action_dataset[5]

        assert isinstance(observation, Observation), "Returned object must be Observation"

        # Images may or may not exist depending on dataset variant
        if "observation.images.wrist" in raw_lerobot_dataset[0]:
            assert isinstance(observation.images, dict)
            assert "wrist" in observation.images
        else:
            assert observation.images == {}

        # Episode index should always be present
        assert observation.episode_index == 0

    def test_from_lerobot_factory_method(self, raw_lerobot_dataset):
        """Tests the `from_lerobot` static method with a mock instance."""
        action_dataset = _LeRobotDatasetAdapter.from_lerobot(raw_lerobot_dataset)

        assert action_dataset._lerobot_dataset is raw_lerobot_dataset

        observation = action_dataset[0]
        raw_item = raw_lerobot_dataset[0]

        # Action may be continuous or discrete
        if "action" in raw_item:
            assert torch.equal(observation.action, raw_item["action"])
        elif "action.continuous" in raw_item:
            assert torch.equal(observation.action["continuous"], raw_item["action.continuous"])
        else:
            raise AssertionError("No recognizable action field in mock dataset")

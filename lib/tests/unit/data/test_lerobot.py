# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for lerobot dataset using a mock to avoid ffmpeg/network dependencies."""

from action_trainer.data import ActionDataset, LeRobotActionDataset, Observation
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
            "observation.images.image": torch.randn(3, 64, 64),
            "observation.state": torch.randn(8),
            "action": torch.randn(7),
            "episode_index": torch.tensor(0),
            "frame_index": torch.tensor(idx),
            "index": torch.tensor(idx),
            "task": "pusht",
            "task_index": torch.tensor(0),
            "timestamp": torch.tensor(float(idx) / 10.0),
        }


class TestLeRobotActionDataset:
    """Groups tests for the LeRobotActionDataset wrapper, using a mock dataset."""

    @pytest.fixture(scope="class")
    def raw_lerobot_dataset(self) -> FakeLeRobotDataset:
        """
        Fixture to provide a MOCK LeRobotDataset instance,
        loaded only once per class.
        """
        return FakeLeRobotDataset()

    def test_initialization(self, monkeypatch):
        """
        Tests that LeRobotActionDataset initializes correctly by patching
        the real LeRobotDataset with our mock.
        """
        # Replace the real LeRobotDataset with our mock at the point of use
        monkeypatch.setattr(
            "action_trainer.data.lerobot.LeRobotDataset", FakeLeRobotDataset
        )

        # This now calls MockLeRobotDataset's constructor instead of the real one
        dataset = LeRobotActionDataset(repo_id="any/repo", episodes=[0])

        assert isinstance(dataset, ActionDataset)
        # Check that the internal attribute is an instance of our mock
        assert isinstance(dataset._lerobot_dataset, FakeLeRobotDataset)
        assert len(dataset) > 0

    def test_len_delegation(self, raw_lerobot_dataset: FakeLeRobotDataset):
        """Tests that __len__ correctly delegates to the mock dataset."""
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)
        assert len(action_dataset) == len(raw_lerobot_dataset)
        assert len(action_dataset) == 150 # Check against the mock's fixed length

    def test_getitem_returns_observation(self, raw_lerobot_dataset: FakeLeRobotDataset):
        """Tests that __getitem__ returns a correctly formatted Observation object."""
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)
        observation = action_dataset[5]  # Get an arbitrary item

        assert isinstance(observation, Observation)
        assert isinstance(observation.images, dict)
        assert "image" in observation.images
        assert isinstance(observation.state, torch.Tensor)
        assert observation.episode_index == 0

    def test_from_lerobot_factory_method(self, raw_lerobot_dataset: FakeLeRobotDataset):
        """Tests the `from_lerobot` static method with a mock instance."""
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)

        # Check that the internal dataset is the *exact same object*
        assert action_dataset._lerobot_dataset is raw_lerobot_dataset

        # Check that item access and conversion work
        observation = action_dataset[0]
        raw_item = raw_lerobot_dataset[0]
        assert torch.equal(observation.action, raw_item["action"])

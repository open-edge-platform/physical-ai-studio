# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for lerobot dataset"""

from action_trainer.data import ActionDataset, LeRobotActionDataset, Observation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
import pytest


class TestLeRobotActionDataset:
    """Groups tests for the LeRobotActionDataset wrapper."""

    # TODO: Implement a lerobot dataset rather than relyin on huggingface
    REPO_ID = "lerobot/pusht"

    @pytest.fixture(scope="class")
    def raw_lerobot_dataset(self) -> LeRobotDataset:
        """Fixture to provide a raw LeRobotDataset instance, loaded only once per class."""
        return LeRobotDataset(repo_id=self.REPO_ID, episodes=[0])

    def test_initialization(self):
        """
        Tests that LeRobotActionDataset initializes correctly and wraps a LeRobotDataset.
        """
        dataset = LeRobotActionDataset(repo_id=self.REPO_ID, episodes=[0])
        assert isinstance(dataset, ActionDataset)
        # Check that the internal attribute is a LeRobotDataset instance
        assert isinstance(dataset._lerobot_dataset, LeRobotDataset) # noqa: SLF001
        assert len(dataset) > 0

    def test_len_delegation(self, raw_lerobot_dataset: LeRobotDataset):
        """
        Tests that the __len__ method correctly delegates to the wrapped dataset.
        """
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)
        assert len(action_dataset) == len(raw_lerobot_dataset)

    def test_getitem_returns_observation(self, raw_lerobot_dataset: LeRobotDataset):
        """
        Tests that __getitem__ returns a correctly formatted Observation object.
        """
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)
        observation = action_dataset[5]  # Get an arbitrary item

        assert isinstance(observation, Observation)
        assert isinstance(observation.images, dict)
        assert isinstance(observation.state, torch.Tensor)
        assert observation.episode_index == 0

    def test_from_lerobot_factory_method(self, raw_lerobot_dataset: LeRobotDataset):
        """
        Tests the `from_lerobot` static method to ensure it correctly wraps an
        existing LeRobotDataset instance without re-initializing it.
        """
        action_dataset = LeRobotActionDataset.from_lerobot(raw_lerobot_dataset)

        # Check that the internal dataset is the *exact same object*
        assert action_dataset._lerobot_dataset is raw_lerobot_dataset # noqa: SLF001

        # Check that the wrapper works as expected
        assert len(action_dataset) == len(raw_lerobot_dataset)

        # Check that item access and conversion work
        observation = action_dataset[0]
        raw_item = raw_lerobot_dataset[0]
        assert torch.equal(observation.action, raw_item["action"])

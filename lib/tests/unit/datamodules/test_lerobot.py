# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test for lerobot datamodule"""

import pytest
import torch
from action_trainer.data import ActionDataset, ActionDataModule, Observation


class FakeActionDataset(ActionDataset):
    """A fake ActionDataset for testing purposes."""
    def __init__(self, length: int = 100):
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Observation:
        # The Observation must be initialized with all required fields.
        return Observation(
            action=torch.randn(4),
            task="stack blocks",
            index=idx,
        )

    @property
    def features(self) -> dict:
        """
        Raw dataset features.
        """
        return {}

    @property
    def action_features(self) -> dict:
        """
        Action features from the dataset.
        """
        return {}

    @property
    def fps(self) -> int:
        """
        Frames per second of the dataset.
        """
        return 30

    @property
    def tolerance_s(self) -> float:
        """
        Tolerance to keep delta timestamps in sync with fps.
        """
        return 0.1

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """
        Exposes delta_indices from the dataset.
        """
        return {"test": [1, 2]}

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]):
        """
        Allows setting delta_indices on the dataset.
        """
        pass



# TODO: Add tests for gym envs concat
class TestActionDataModule:
    """Groups all tests for the ActionDataModule."""

    @pytest.fixture
    def mock_train_dataset(self) -> FakeActionDataset:
        """Provides a fake training dataset instance."""
        return FakeActionDataset(length=128)

    def test_initialization(self, mock_train_dataset: FakeActionDataset):
        """Tests if the DataModule initializes attributes correctly."""
        dm = ActionDataModule(
            train_dataset=mock_train_dataset,
            train_batch_size=32
        )
        assert dm.train_dataset is mock_train_dataset
        assert dm.train_batch_size == 32
        assert dm.eval_dataset is None
        assert dm.test_dataset is None

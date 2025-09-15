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

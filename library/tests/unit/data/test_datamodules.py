# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DataModule."""

from __future__ import annotations


class TestDataModuleValidation:
    """Tests for DataModule validation functionality."""

    def test_collate_gym_observation(self):
        """Test that gym collate function creates GymObservation."""
        from getiaction.data.datamodules import _collate_gym_observation
        from getiaction.data.observation import GymObservation
        from getiaction.gyms import PushTGym

        gym = PushTGym()
        batch = [gym]  # Simulates batch from DataLoader

        result = _collate_gym_observation(batch)

        assert isinstance(result, GymObservation)
        assert result.env is gym

    def test_val_dataloader_structure(self, dummy_datamodule):
        """Test that DataModule.val_dataloader returns correct structure."""
        from getiaction.data.observation import GymObservation

        dummy_datamodule.setup(stage="fit")
        val_loader = dummy_datamodule.val_dataloader()

        # Should have 2 batches (num_rollouts_val=2 from fixture)
        assert len(val_loader) == 2

        # Get one batch
        batch = next(iter(val_loader))

        # Should be a GymObservation
        assert isinstance(batch, GymObservation)
        assert batch.env is not None

    def test_val_dataloader_with_multiple_gyms(self, dummy_dataset):
        """Test that val_dataloader works with list of gyms."""
        from getiaction.data import DataModule
        from getiaction.gyms import PushTGym

        gym1 = PushTGym()
        gym2 = PushTGym()
        datamodule = DataModule(
            train_dataset=dummy_dataset(num_samples=20),
            train_batch_size=4,
            val_gyms=[gym1, gym2],
            num_rollouts_val=2,
        )

        datamodule.setup(stage="fit")

        val_loader = datamodule.val_dataloader()

        # Should have 4 batches (2 gyms * 2 rollouts each)
        assert len(val_loader) == 4

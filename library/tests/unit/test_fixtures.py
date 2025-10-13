# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared fixtures."""

from __future__ import annotations


class TestDummyDatasetFixture:
    """Tests for the dummy_dataset fixture."""

    def test_dummy_dataset_creates_instances(self, dummy_dataset):
        """Test that dummy_dataset fixture creates proper dataset instances."""
        dataset = dummy_dataset(num_samples=10)

        assert len(dataset) == 10

    def test_dummy_dataset_returns_observations(self, dummy_dataset):
        """Test that dummy dataset returns Observation objects."""
        from getiaction.data.observation import Observation

        dataset = dummy_dataset(num_samples=5)
        sample = dataset[0]

        assert isinstance(sample, Observation)
        assert sample.action is not None
        assert sample.state is not None
        assert sample.images is not None

    def test_dummy_dataset_different_sizes(self, dummy_dataset):
        """Test creating datasets with different sizes."""
        small_dataset = dummy_dataset(num_samples=5)
        large_dataset = dummy_dataset(num_samples=100)

        assert len(small_dataset) == 5
        assert len(large_dataset) == 100


class TestDummyLeRobotDatasetFixture:
    """Tests for the dummy_lerobot_dataset fixture."""

    def test_dummy_lerobot_dataset_structure(self, dummy_lerobot_dataset):
        """Test that dummy LeRobot dataset has correct structure."""
        dataset = dummy_lerobot_dataset(num_samples=50)

        assert len(dataset) == 50
        assert hasattr(dataset, "episode_data_index")
        assert hasattr(dataset, "features")
        assert hasattr(dataset, "meta")

    def test_dummy_lerobot_dataset_returns_dict(self, dummy_lerobot_dataset):
        """Test that dummy LeRobot dataset returns dict batches."""
        dataset = dummy_lerobot_dataset(num_samples=20)
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "observation.state" in sample
        assert "observation.image" in sample
        assert "action" in sample
        assert "episode_index" in sample

    def test_dummy_lerobot_dataset_features(self, dummy_lerobot_dataset):
        """Test that dummy LeRobot dataset has features property."""
        dataset = dummy_lerobot_dataset(num_samples=10)
        features = dataset.features

        assert "observation.state" in features
        assert "observation.image" in features
        assert "action" in features

    def test_dummy_lerobot_dataset_metadata(self, dummy_lerobot_dataset):
        """Test that dummy LeRobot dataset has meta property."""
        dataset = dummy_lerobot_dataset(num_samples=10)
        meta = dataset.meta

        assert "robot_type" in meta
        assert "fps" in meta
        assert meta["robot_type"] == "dummy_robot"


class TestDummyDataModuleFixture:
    """Tests for the dummy_datamodule fixture."""

    def test_dummy_datamodule_has_dataloaders(self, dummy_datamodule):
        """Test that dummy datamodule can create dataloaders."""
        dummy_datamodule.setup(stage="fit")

        train_loader = dummy_datamodule.train_dataloader()
        val_loader = dummy_datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

    def test_dummy_datamodule_train_loader(self, dummy_datamodule):
        """Test that train loader returns correct batches."""
        from getiaction.data.observation import Observation

        dummy_datamodule.setup(stage="fit")
        train_loader = dummy_datamodule.train_dataloader()

        batch = next(iter(train_loader))
        assert isinstance(batch, Observation)

    def test_dummy_datamodule_val_loader(self, dummy_datamodule):
        """Test that val loader returns GymObservation."""
        from getiaction.data.observation import GymObservation

        dummy_datamodule.setup(stage="fit")
        val_loader = dummy_datamodule.val_dataloader()

        batch = next(iter(val_loader))
        assert isinstance(batch, GymObservation)


class TestDummyLeRobotDataModuleFixture:
    """Tests for the dummy_lerobot_datamodule fixture."""

    def test_dummy_lerobot_datamodule_creation(self, dummy_lerobot_datamodule):
        """Test that dummy LeRobot-style datamodule can be created."""
        assert dummy_lerobot_datamodule is not None

    def test_dummy_lerobot_datamodule_has_dataloaders(self, dummy_lerobot_datamodule):
        """Test that dummy datamodule can create dataloaders."""
        dummy_lerobot_datamodule.setup(stage="fit")

        train_loader = dummy_lerobot_datamodule.train_dataloader()
        val_loader = dummy_lerobot_datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

    def test_dummy_lerobot_dataset_used_directly(self, dummy_lerobot_dataset):
        """Test that dummy LeRobot dataset can be used directly without DataModule."""
        dataset = dummy_lerobot_dataset(num_samples=50)

        # Test direct access - useful for testing LeRobot policies
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "action" in sample
        assert "observation.state" in sample or "observation.image" in sample

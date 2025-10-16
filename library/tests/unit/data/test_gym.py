# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GymDataset."""

from __future__ import annotations


class TestGymDataset:
    """Tests for GymDataset."""

    def test_gym_dataset_returns_gym(self):
        """Test that GymDataset returns Gym objects."""
        from getiaction.data.gym import GymDataset
        from getiaction.gyms import PushTGym

        gym = PushTGym()
        dataset = GymDataset(env=gym, num_rollouts=5)

        assert len(dataset) == 5
        item = dataset[0]
        assert item is gym

    def test_gym_dataset_length(self):
        """Test that GymDataset has correct length."""
        from getiaction.data.gym import GymDataset
        from getiaction.gyms import PushTGym

        gym = PushTGym()
        dataset = GymDataset(env=gym, num_rollouts=10)

        assert len(dataset) == 10

    def test_gym_dataset_indexing(self):
        """Test that GymDataset indexing returns same gym."""
        from getiaction.data.gym import GymDataset
        from getiaction.gyms import PushTGym

        gym = PushTGym()
        dataset = GymDataset(env=gym, num_rollouts=3)

        # All indices should return the same gym instance
        assert dataset[0] is gym
        assert dataset[1] is gym
        assert dataset[2] is gym

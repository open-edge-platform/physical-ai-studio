# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataset for Gyms."""

from __future__ import annotations

from torch.utils.data import Dataset

from getiaction.gyms import Gym


class GymDataset(Dataset[Gym]):
    """A dataset wrapper for Gym environments.

    This class allows a Gym environment to be treated as a dataset,
    where each item corresponds to a single rollout of the environment.
    """

    def __init__(self, env: Gym, num_rollouts: int) -> None:
        """Initialize the GymDataset.

        Args:
            env (Gym): The Gym environment to wrap as a dataset.
            num_rollouts (int): The number of rollouts to include in the dataset.
        """
        self.env: Gym = env
        self.num_rollouts: int = num_rollouts

    def __len__(self) -> int:
        """Return the number of rollouts in the dataset.

        Returns:
            int: Number of rollouts, derived from the environment.
        """
        return self.num_rollouts

    def __getitem__(self, index: int) -> Gym:
        """Retrieve a specific rollout of the environment.

        This resets the environment with a seed corresponding to the index.

        Args:
            index (int): Index of the rollout.

        Returns:
            Gym: The environment after being reset for this rollout.
        """
        self.env.reset(seed=index)
        return self.env

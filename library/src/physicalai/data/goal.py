# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for datasets.

A goal shows the state a policy should reach (e.g. the final frame of a demonstration).
`GoalConditionedDataset` attaches its images to every sample as ``Observation.goal_images``;
subclasses decide where the goal comes from.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from physicalai.data.dataset import Dataset

if TYPE_CHECKING:
    import numpy as np
    import torch

    from physicalai.data.observation import Feature, Observation


class TaskGoalStore(Protocol):
    """Provides rollout goal images for task descriptions."""

    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Return the goal images for a task description."""
        ...


class GoalConditionedDataset(Dataset):
    """Base wrapper that attaches a goal to every `Observation` of a dataset.

    Subclasses implement `get_goal` for training samples and `get_task_goal` for rollouts
    (used by `physicalai.gyms.GoalConditionedGym`). Everything else behaves like the wrapped
    dataset.

    Goal images are laid out like ``Observation.images``. They are not added to
    ``observation_features``, so policies that do not use goals are unaffected. Policies
    that do should normalise ``goal_images`` with the same statistics as ``images``.
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize the wrapper.

        Args:
            dataset: Dataset whose samples get a goal attached.
        """
        super().__init__()
        self.dataset = dataset

    @abstractmethod
    def get_goal(
        self,
        observation: Observation,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Return the goal images for a sample of the wrapped dataset."""

    @abstractmethod
    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Return the fixed goal images for a task description, e.g. ``Observation.task`` from a gym."""

    def __len__(self) -> int:
        """Returns the number of samples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Observation:
        """Returns the wrapped sample with its goal attached."""
        obs = self.dataset[idx]
        obs.goal_images = self.get_goal(obs)
        return obs

    @property
    def raw_features(self) -> dict:
        """Raw features of the wrapped dataset."""
        return self.dataset.raw_features

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Observation features of the wrapped dataset."""
        return self.dataset.observation_features

    @property
    def action_features(self) -> dict[str, Feature]:
        """Action features of the wrapped dataset."""
        return self.dataset.action_features

    @property
    def fps(self) -> int:
        """Frames per second of the wrapped dataset."""
        return self.dataset.fps

    @property
    def tolerance_s(self) -> float:
        """Timestamp tolerance of the wrapped dataset."""
        return self.dataset.tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Delta indices of the wrapped dataset."""
        return self.dataset.delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self.dataset.delta_indices = indices


__all__ = ["GoalConditionedDataset", "TaskGoalStore"]

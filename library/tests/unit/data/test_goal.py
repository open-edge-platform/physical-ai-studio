# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic GoalConditionedDataset base class."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physicalai.data import Dataset, GoalConditionedDataset, Observation


class EpisodeDataset(Dataset):
    """Two frames per episode."""

    def __init__(self) -> None:
        self._delta_indices: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> Observation:
        return Observation(state=torch.zeros(2), episode_index=torch.tensor(idx // 2), extra={"keep": 1})

    @property
    def raw_features(self) -> dict:
        return {"raw": 1}

    @property
    def observation_features(self) -> dict:
        return {"obs": 1}

    @property
    def action_features(self) -> dict:
        return {"act": 1}

    @property
    def fps(self) -> int:
        return 10

    @property
    def tolerance_s(self) -> float:
        return 1e-4

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        return self._delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self._delta_indices = indices


class EpisodeGoalDataset(GoalConditionedDataset):
    """Goal filled with the episode index."""

    def get_goal(
        self, observation: Observation,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        return {"top": torch.full((3, 2, 2), float(observation.episode_index))}

    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        return {"top": torch.full((3, 2, 2), float(len(task)))}


def test_base_class_is_abstract():
    with pytest.raises(TypeError, match="get_task_goal"):
        GoalConditionedDataset(EpisodeDataset())  # type: ignore[abstract]


def test_attaches_goal_from_subclass():
    obs = EpisodeGoalDataset(EpisodeDataset())[5]  # episode 2

    assert torch.equal(obs.goal_images["top"], torch.full((3, 2, 2), 2.0))
    assert obs.extra == {"keep": 1}


def test_behaves_like_the_wrapped_dataset():
    dataset = EpisodeGoalDataset(EpisodeDataset())
    dataset.delta_indices = {"action": [0, 1]}

    assert len(dataset) == 6
    assert dataset.raw_features == {"raw": 1}
    assert dataset.observation_features == {"obs": 1}
    assert dataset.action_features == {"act": 1}
    assert dataset.fps == 10
    assert dataset.tolerance_s == 1e-4
    assert dataset.dataset.delta_indices == {"action": [0, 1]}

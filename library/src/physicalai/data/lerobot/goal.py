# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Goal conditioning for LeRobot datasets using precomputed goal row indices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from lightning_utilities import module_available

from physicalai.data.goal import GoalConditionedDataset

from .dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    import numpy as np
    import torch
    from pandas import DataFrame

    from physicalai.data.observation import Observation

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

GoalSource = Literal["task", "episode"]
EpisodeGoalFrames = dict[int, int] | str | Path
TaskGoalEpisodes = dict[str, int]


def _normalize_task(task: str) -> str:
    return " ".join(task.lower().split()).rstrip(".")


def _build_task_indices(tasks: DataFrame) -> dict[str, int]:
    task_indices: dict[str, int] = {}
    for task, index in zip(tasks.index, tasks["task_index"], strict=True):
        normalized = _normalize_task(str(task))
        task_index = int(index)
        if normalized in task_indices and task_indices[normalized] != task_index:
            msg = f"Task descriptions normalize to the same key '{normalized}'."
            raise ValueError(msg)
        task_indices[normalized] = task_index
    return task_indices


def _metadata_episodes(dataset: LeRobotDataset) -> list[dict[str, Any]]:
    episodes = cast("list[dict[str, Any]]", list(dataset.meta.episodes))
    return sorted(episodes, key=lambda item: int(item["dataset_from_index"]))


@dataclass(frozen=True)
class _Episode:
    """Selected episode expressed in split-local dataset rows."""

    index: int
    task_index: int
    start: int
    length: int

    def goal_row(self, offset: int) -> int:
        """Resolve an episode-relative frame offset to a split-local row.

        Returns:
            The split-local dataset row.

        Raises:
            ValueError: If the offset falls outside the episode.
        """
        frame = offset + self.length if offset < 0 else offset
        if frame < 0 or frame >= self.length:
            msg = f"Goal frame offset {offset} is outside episode {self.index}, which contains {self.length} frames."
            raise ValueError(msg)
        return self.start + frame


def _parse_episode_goal_frames(value: EpisodeGoalFrames | None) -> dict[int, int]:
    if value is None:
        return {}

    from_json = False
    if isinstance(value, (str, Path)):
        from_json = True
        path = Path(value)
        if path.suffix.lower() != ".json":
            msg = f"Episode goal frame file must be JSON, got '{path.suffix}'."
            raise ValueError(msg)
        with path.open(encoding="utf-8") as file:
            value = json.load(file)

    if not isinstance(value, dict):
        msg = "Episode goal frames must be a mapping of episode index to integer frame offset."
        raise ValueError(msg)  # noqa: TRY004

    frames: dict[int, int] = {}
    for raw_episode_index, offset in value.items():
        if from_json:
            if not isinstance(raw_episode_index, str):
                msg = f"Episode goal frame key {raw_episode_index!r} is not an integer string."
                raise ValueError(msg)
            try:
                episode_index = int(raw_episode_index)
            except ValueError as exc:
                msg = f"Episode goal frame key '{raw_episode_index}' is not an integer episode index."
                raise ValueError(msg) from exc
            if str(episode_index) != raw_episode_index:
                msg = f"Episode goal frame key '{raw_episode_index}' is not a canonical integer string."
                raise ValueError(msg)
        else:
            if type(raw_episode_index) is not int:
                msg = f"Episode goal frame key {raw_episode_index!r} must be an integer episode index."
                raise ValueError(msg)
            episode_index = raw_episode_index

        if type(offset) is not int:
            msg = f"Episode goal frame offset for episode {episode_index} must be an integer, got {offset!r}."
            raise ValueError(msg)
        frames[episode_index] = offset
    return frames


def _validate_goal_frames(metadata: list[dict[str, Any]], frames: dict[int, int]) -> None:
    """Validate configured offsets against the complete dataset metadata.

    Raises:
        KeyError: If an offset refers to an unknown episode.
        ValueError: If an offset falls outside its episode.
    """
    metadata_by_id = {int(episode["episode_index"]): episode for episode in metadata}
    unknown = set(frames) - set(metadata_by_id)
    if unknown:
        msg = f"Episode goal frames reference unknown episodes: {sorted(unknown)}"
        raise KeyError(msg)

    for episode_index, offset in frames.items():
        episode = metadata_by_id[episode_index]
        length = int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        frame = offset + length if offset < 0 else offset
        if length <= 0 or frame < 0 or frame >= length:
            msg = f"Goal frame offset {offset} is outside episode {episode_index}, which contains {length} frames."
            raise ValueError(msg)


def _selected_episodes(dataset: LeRobotDataset, metadata: list[dict[str, Any]]) -> list[_Episode]:
    """Describe selected episodes using split-local row positions.

    Returns:
        Selected episodes in dataset order.

    Raises:
        ValueError: If selection metadata does not match the dataset rows.
    """
    known_ids = {int(episode["episode_index"]) for episode in metadata}
    selected_ids = set(dataset.episodes) if dataset.episodes is not None else known_ids
    missing = selected_ids - known_ids
    if missing:
        msg = f"Dataset metadata is missing selected episodes: {sorted(missing)}"
        raise ValueError(msg)
    if not selected_ids:
        msg = "Cannot index goals for a dataset with no selected episodes."
        raise ValueError(msg)

    selected: list[_Episode] = []
    relative_start = 0
    for item in metadata:
        episode_index = int(item["episode_index"])
        if episode_index not in selected_ids:
            continue
        length = int(item["dataset_to_index"]) - int(item["dataset_from_index"])
        if length <= 0:
            msg = f"Episode {episode_index} has invalid frame bounds."
            raise ValueError(msg)
        selected.append(
            _Episode(
                index=episode_index,
                task_index=int(dataset.get_raw_item(relative_start)["task_index"]),
                start=relative_start,
                length=length,
            ),
        )
        relative_start += length

    if relative_start != len(dataset):
        msg = f"Indexed {relative_start} rows for a dataset containing {len(dataset)} samples."
        raise ValueError(msg)
    return selected


class LeRobotGoalConditionedDataset(GoalConditionedDataset):
    """LeRobot wrapper that selects goal rows by task or episode."""

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        dataset: LeRobotDataset | _LeRobotDatasetAdapter,
        goal_source: GoalSource | None = None,
        *,
        episode_goal_frames: EpisodeGoalFrames | None = None,
        task_goal_episodes: TaskGoalEpisodes | None = None,
        task_goal_source: LeRobotGoalConditionedDataset | None = None,
    ) -> None:
        """Select one goal row per episode and task without decoding images.

        Args:
            dataset: LeRobot dataset or adapter to wrap.
            goal_source: Whether observations use task-level or episode-level goals. Defaults to ``"task"`` when
                neither goal selection mapping is provided; otherwise, it must be specified.
            episode_goal_frames: Optional mapping from episode indices to goal frame offsets, or a JSON file
                containing that mapping. Offsets may be negative; unspecified episodes use their final frame.
            task_goal_episodes: Optional mapping from task descriptions to representative episode indices.
                Episodes outside this dataset split use the first local episode for the task.
            task_goal_source: Optional training dataset wrapper whose task goals should be reused, typically
                for validation or test datasets.

        Raises:
            ValueError: If source selection or episode metadata is invalid.
        """
        # if either episode_goal_frames or task_goal_episodes is provided, goal_source must be explicitly set
        selection_configured = episode_goal_frames is not None or task_goal_episodes is not None
        if goal_source is None and selection_configured:
            msg = "'episode_goal_frames' and 'task_goal_episodes' require an explicit goal_source."
            raise ValueError(msg)

        # default goal source to task
        self.goal_source: GoalSource = goal_source or "task"

        if self.goal_source not in {"task", "episode"}:
            msg = f"Unknown goal source '{self.goal_source}', expected 'task' or 'episode'."
            raise ValueError(msg)

        # wrap the raw dataset in an adapter if necessary, to extend to non-lerobot datasets
        # we should modify it here
        if not isinstance(dataset, _LeRobotDatasetAdapter):
            dataset = _LeRobotDatasetAdapter.from_lerobot(dataset)
        super().__init__(dataset)
        raw_dataset = dataset._lerobot_dataset  # noqa: SLF001
        if getattr(raw_dataset, "delta_timestamps", None):
            msg = "Goal conditioning requires delta_timestamps=None to select a single goal frame."
            raise ValueError(msg)

        # build task indices and parse episode goal frames
        self._task_indices = _build_task_indices(raw_dataset.meta.tasks)
        metadata = _metadata_episodes(raw_dataset)
        configured_frames = _parse_episode_goal_frames(episode_goal_frames)
        _validate_goal_frames(metadata, configured_frames)
        episodes = _selected_episodes(raw_dataset, metadata)

        # configure task and episode goal indices based on the provided sources and episodes
        self._task_goal_source = task_goal_source
        if task_goal_source is not None and self.goal_source != "task":
            msg = "A training task-goal source can only be used with goal_source='task'."
            raise ValueError(msg)
        if task_goal_source is not None and task_goal_episodes is not None:
            msg = "Configure task goal episodes on the training task-goal source."
            raise ValueError(msg)
        self._episode_goal_indices = (
            {episode.index: episode.goal_row(configured_frames.get(episode.index, -1)) for episode in episodes}
            if task_goal_source is None or self.goal_source == "episode"
            else {}
        )
        self._task_goal_indices: dict[int, int] = {}
        if task_goal_source is not None:
            training_task_indices = task_goal_source._task_goal_indices.keys()  # noqa: SLF001
            missing_tasks = {episode.task_index for episode in episodes} - training_task_indices
            if missing_tasks:
                msg = f"Validation tasks {sorted(missing_tasks)} have no training goal."
                raise ValueError(msg)
        else:
            for episode in episodes:
                self._task_goal_indices.setdefault(episode.task_index, self._episode_goal_indices[episode.index])

        # configure task goal indices based on the provided task goal episodes
        if task_goal_episodes is not None:
            episodes_by_id = {episode.index: episode for episode in episodes}
            for task, episode_index in task_goal_episodes.items():
                if type(episode_index) is not int:
                    msg = f"Task '{task}' goal episode must be an integer, got {episode_index!r}."
                    raise ValueError(msg)
                task_index = self._task_index(task)
                # A representative outside this split falls back to the first local episode.
                if episode_index not in episodes_by_id:
                    continue
                episode = episodes_by_id[episode_index]
                if episode.task_index != task_index:
                    msg = f"Task '{task}' goal episode {episode_index} does not belong to that task."
                    raise ValueError(msg)
                self._task_goal_indices[task_index] = self._episode_goal_indices[episode_index]

    def _task_index(self, task: str) -> int:
        normalized = _normalize_task(task)
        if normalized not in self._task_indices:
            msg = f"Task '{task}' is not in the dataset. Dataset tasks: {sorted(self._task_indices)}"
            raise KeyError(msg)
        return self._task_indices[normalized]

    def get_goal(
        self,
        observation: Observation,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Read the goal selected for an observation through normal dataset access.

        Returns:
            Goal images from the configured task or episode frame.

        Raises:
            KeyError: If the observation has no goal in this split.
        """
        if self.goal_source == "episode":
            episode_index = int(observation.episode_index)  # type: ignore[arg-type]
            if episode_index not in self._episode_goal_indices:
                msg = f"No goal for episode index {episode_index} in the wrapped dataset."
                raise KeyError(msg)
            goal_index = self._episode_goal_indices[episode_index]
        else:
            task_index = int(observation.task_index)  # type: ignore[arg-type]
            if self._task_goal_source is not None:
                return self._task_goal_source._get_task_goal_by_index(task_index)  # noqa: SLF001
            return self._get_task_goal_by_index(task_index)
        return self.dataset[goal_index].images

    def _get_task_goal_by_index(
        self,
        task_index: int,
    ) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        if task_index not in self._task_goal_indices:
            msg = f"No goal for task index {task_index} in the wrapped dataset."
            raise KeyError(msg)
        return self.dataset[self._task_goal_indices[task_index]].images

    def get_task_goal(self, task: str) -> dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None:
        """Read the configured representative goal for a task through normal dataset access.

        Returns:
            Representative task goal images, using the configured task-goal source.
        """
        task_index = self._task_index(task)
        if self._task_goal_source is not None:
            return self._task_goal_source._get_task_goal_by_index(task_index)  # noqa: SLF001
        return self._get_task_goal_by_index(task_index)


__all__ = ["EpisodeGoalFrames", "GoalSource", "LeRobotGoalConditionedDataset", "TaskGoalEpisodes"]

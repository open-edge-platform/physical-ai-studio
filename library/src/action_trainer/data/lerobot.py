# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobotDataset standard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_trainer.data import DataModule, Dataset, Observation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import torch


def _collect_field(
    item: dict,
    base_key: str,
    prefix: str | None = None,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor | None, set[str]]:
    """Collect fields from `item` based on `base_key` and `prefix`.

    Returns:
        - Either a single tensor, a dict, or None
        - The set of keys that were consumed
    """
    if prefix is None:
        prefix = base_key + "."

    collected: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()

    # exact single key
    if base_key in item:
        collected[base_key] = item[base_key]
        used_keys.add(base_key)

    # prefixed subkeys
    for key, value in item.items():
        if key.startswith(prefix):
            subkey = key.split(prefix, 1)[1]
            collected[subkey] = value
            used_keys.add(key)

    if not collected:
        return None, used_keys
    if len(collected) == 1 and base_key in collected:
        return collected[base_key], used_keys
    return collected, used_keys


def _convert_lerobot_item_to_observation(lerobot_item: dict) -> Observation:
    """Convert item from lerobot to our internal Observation format."""
    required_keys = [
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "timestamp",
    ]
    lerobot_item_keys = lerobot_item.keys()
    for key in required_keys:
        assert key in lerobot_item_keys, f"Missing required key: {key}. Available keys: {lerobot_item_keys}"

    assert any(k.startswith("observation") for k in lerobot_item_keys), (
        f"Sample must contain some form of observation. Sample keys {lerobot_item_keys}"
    )
    assert any(k.startswith("action") for k in lerobot_item_keys), (
        f"Sample must contain an action. Sample keys {lerobot_item_keys}"
    )

    used_keys: set[str] = set()

    # Observation images
    images, used = _collect_field(lerobot_item, "observation.image", "observation.images.")
    used_keys |= used

    # Observation states
    state, used = _collect_field(lerobot_item, "observation.state", "observation.state.")
    used_keys |= used

    # Actions
    action, used = _collect_field(lerobot_item, "action", "action.")
    used_keys |= used

    # Tasks
    task, used = _collect_field(lerobot_item, "task", "task.")
    used_keys |= used

    # Extra keys
    reserved = set(required_keys) | used_keys
    extra = {k: v for k, v in lerobot_item.items() if k not in reserved}

    return Observation(
        images=images if images is not None else {},
        state=state if state is not None else None,
        action=action if action is not None else None,
        task=task if task is not None else None,
        episode_index=lerobot_item["episode_index"],
        frame_index=lerobot_item["frame_index"],
        index=lerobot_item["index"],
        task_index=lerobot_item["task_index"],
        timestamp=lerobot_item["timestamp"],
        extra=extra,
    )


# NOTE: Eventually we will need properties from action and lerobot datasets
class LeRobotDatasetWrapper(Dataset):
    """A wrapper class that enables using a LeRobotDataset for action training.

    This class uses the **composition** pattern by holding a `LeRobotDataset` instance
    and delegating all data loading operations to it. This allows adding action-specific
    logic without modifying the original `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        *,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float], str] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> None:
        """Initialize a LeRobotDatasetWrapper.

        This wrapper initializes an internal `LeRobotDataset` using the provided
        configuration and exposes the same dataset interface for action training.

        Args:
            repo_id (str): Repository ID of the LeRobot dataset.
            root (str | Path | None, optional): Local root directory to cache dataset files. Defaults to None.
            episodes (list[int] | None, optional): Specific episode indices to include. Defaults to None.
            image_transforms (Callable | None, optional): Transformations to apply to images. Defaults to None.
            delta_timestamps (dict[str, list[float]] | None, optional): Mapping of signal keys to timestamp offsets.
            Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds when aligning timestamps. Defaults to 1e-4.
            revision (str | None, optional): Dataset version or branch to use. Defaults to None.
            force_cache_sync (bool, optional): If True, forces synchronization of the dataset cache. Defaults to False.
            download_videos (bool, optional): Whether to download associated videos. Defaults to True.
            video_backend (str | None, optional): Backend to use for video decoding. Defaults to None.
            batch_encoding_size (int, optional): Number of samples per encoded batch. Defaults to 1.
        """
        super().__init__()

        # All arguments are passed
        self._lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )

    def __len__(self) -> int:
        return len(self._lerobot_dataset)

    def __getitem__(self, idx: int) -> Observation:
        return _convert_lerobot_item_to_observation(self._lerobot_dataset[idx])

    @staticmethod
    def from_lerobot(lerobot_dataset: LeRobotDataset) -> LeRobotDatasetWrapper:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.

        Returns:
            LeRobotActionDataset: A new LeRobotActionDataset instance that uses the provided dataset.
        """
        instance = LeRobotDatasetWrapper.__new__(LeRobotDatasetWrapper)
        # Bypassing __init__ to set the internal dataset
        instance._lerobot_dataset = lerobot_dataset  # noqa: SLF001
        return instance

    @property
    def features(self) -> dict[str, dict[Any, Any]]:
        """Raw dataset features."""
        return self._lerobot_dataset.features

    @property
    def action_features(self) -> dict[str, dict[Any, Any]]:
        """Action features from LeRobot dataset."""
        dataset_features = self._lerobot_dataset.features
        return {key: ft for key, ft in dataset_features.items() if key.startswith("action")}

    @property
    def fps(self) -> float:
        """Frames per second of dataset."""
        return self._lerobot_dataset.fps

    @property
    def tolerance_s(self) -> float:
        """Tolerance to keep delta timestamps in sync with fps."""
        return self._lerobot_dataset.tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Expose delta_indices from the dataset."""
        return self._lerobot_dataset.delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Allow setting delta_indices on the dataset."""
        self._lerobot_dataset.delta_indices = indices


class LeRobotDataModule(DataModule):
    """LeRobot-specific Action DataModule."""

    def __init__(
        self,
        train_batch_size: int = 16,
        repo_id: str | None = None,
        dataset: LeRobotDatasetWrapper | LeRobotDataset | None = None,
        # LeRobot Dataset kwargs
        *,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float], str] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        **action_datamodule_kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize a LeRobot-specific Action DataModule.

        This class wraps a `LeRobotDataset` (or `LeRobotDatasetWrapper`) and
        integrates it with the base `ActionDataModule` functionality, providing
        training, evaluation, and test data loaders for imitation learning tasks.

        Args:
            train_batch_size (int, optional): Batch size for the training DataLoader. Defaults to 16.
            repo_id (str | None, optional): Repository ID for the LeRobot dataset.
            Required if `dataset` is not provided.
            dataset (LeRobotDatasetWrapper | LeRobotDataset | None, optional): Pre-initialized dataset instance.
            Defaults to None.
            root (str | Path | None, optional): Local directory for caching dataset files. Defaults to None.
            episodes (list[int] | None, optional): Specific episode indices to include. Defaults to None.
            image_transforms (Callable | None, optional): Transformations to apply to images. Defaults to None.
            delta_timestamps (dict[str, list[float]] | None, optional): Mapping of signal keys to timestamp offsets.
            Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds for aligning timestamps. Defaults to 1e-4.
            revision (str | None, optional): Dataset version or branch to use. Defaults to None.
            force_cache_sync (bool, optional): If True, forces synchronization of the dataset cache. Defaults to False.
            download_videos (bool, optional): Whether to download associated videos. Defaults to True.
            video_backend (str | None, optional): Backend to use for video decoding. Defaults to None.
            batch_encoding_size (int, optional): Number of samples per encoded batch. Defaults to 1.
            **action_datamodule_kwargs: Additional keyword arguments passed to the base `ActionDataModule`.

        Raises:
            ValueError: If neither `repo_id` nor `dataset` is provided, or if the dataset type is invalid.
            TypeError: If `dataset` is not of type `LeRobotDataset` or `LeRobotDatasetWrapper`.
        """
        # if dataset is passed, it is preffered
        if dataset:
            if isinstance(dataset, LeRobotDatasetWrapper):
                train_dataset = dataset
            elif isinstance(dataset, LeRobotDataset):
                train_dataset = LeRobotDatasetWrapper.from_lerobot(dataset)
            else:
                raise TypeError
        elif repo_id:
            # Initialize the LeRobot dataset
            train_dataset = LeRobotDatasetWrapper(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                revision=revision,
                force_cache_sync=force_cache_sync,
                download_videos=download_videos,
                video_backend=video_backend,
                batch_encoding_size=batch_encoding_size,
            )
        else:
            msg = "Must provide either repo_id or a dataset"
            raise ValueError(msg)

        # Pass initialized dataset into the parent class
        super().__init__(
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            **action_datamodule_kwargs,
        )

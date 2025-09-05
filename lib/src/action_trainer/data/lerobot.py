# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
LeRobotDataset standard
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from action_trainer.data import ActionDataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class LeRobotActionDataset(ActionDataset):
    """A wrapper class that enables using a LeRobotDataset for action training.

    This class uses the **composition** pattern, holding a `LeRobotDataset` instance
    and delegating all data loading operations to it. This design allows for
    the addition of action-specific logic without modifying the original
    `LeRobotDataset` class.

    Args:
        repo_id (str): The Hugging Face Hub repository ID.
        root (str | Path | None, optional): Local directory for caching files. Defaults to None.
        episodes (list[int] | None, optional): A list of episode indices to load. Defaults to None.
        image_transforms (Callable | None, optional): A callable to apply to image data. Defaults to None.
        delta_timestamps (dict[str, list[float]] | None, optional): A dict of timestamps. Defaults to None.
        tolerance_s (float, optional): Tolerance in seconds for timestamp synchronization. Defaults to 1e-4.
        revision (str | None, optional): A Git revision ID for the dataset. Defaults to None.
        force_cache_sync (bool, optional): Flag to force synchronization with the Hugging Face Hub. Defaults to False.
        download_videos (bool, optional): Flag to download video files. Defaults to True.
        video_backend (str | None, optional): The video decoding backend to use. Defaults to None.
        batch_encoding_size (int, optional): The number of episodes to encode in a batch. Defaults to 1.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
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

    def __len__(self):
        return len(self._lerobot_dataset)

    # TODO: Should return Observation. Implement interface.
    def __getitem__(self, idx):
        return self._lerobot_dataset[idx]

    @staticmethod
    def from_lerobot(lerobot_dataset: LeRobotDataset) -> LeRobotActionDataset:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.

        Returns:
            LeRobotActionDataset: A new LeRobotActionDataset instance that uses the provided dataset.
        """
        instance = LeRobotActionDataset.__new__(LeRobotActionDataset)
        # Bypassing __init__ to set the internal dataset
        instance._lerobot_dataset = lerobot_dataset  # noqa: SLF001
        return instance

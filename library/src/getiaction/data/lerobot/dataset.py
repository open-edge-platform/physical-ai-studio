# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset adapter for GetiAction compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning_utilities import module_available

from getiaction.data import Dataset

from .converters import FormatConverter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from getiaction.data import Observation

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None


class _LeRobotDatasetAdapter(Dataset):
    """An internal adapter that makes a `LeRobotDataset` compatible with the `getiaction.data.Dataset` interface.

    This adapter class serves two primary purposes:
    1.  **Protocol Compliance**: It wraps the `LeRobotDataset` to ensure it conforms to the
        abstract methods and properties required by the `getiaction.data.Dataset` base class
        (e.g., providing `.features`, `.fps`, etc.).
    2.  **Interface Adaptation**: It transforms the dictionary-based output of `LeRobotDataset.__getitem__`
        into the structured `Observation` dataclass format expected by the training pipeline.

    Note:
        This is an internal implementation detail and is not meant to be used directly by end-users.
        The `LeRobotDataModule` handles the creation and management of this adapter automatically.
    """

    def __init__(
        self,
        *,
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
    ) -> None:
        """Initialize a _LeRobotDatasetAdapter.

        This adapter initializes an internal `LeRobotDataset` using the provided configuration
        and exposes the same dataset interface for action training.

        Args:
            repo_id (str): Repository ID of the LeRobot dataset.
            root (str | Path | None, optional): Local root directory to cache dataset files.
                Defaults to `None`.
            episodes (list[int] | None, optional): Specific episode indices to include.
                Defaults to `None`.
            image_transforms (Callable | None, optional): Transformations to apply to images.
                Defaults to `None`.
            delta_timestamps (dict[str, list[float]] | None, optional): Mapping of signal keys to timestamp offsets.
                Defaults to `None`.
            tolerance_s (float, optional): Tolerance in seconds when aligning timestamps.
                Defaults to `1e-4`.
            revision (str | None, optional): Dataset version or branch to use.
                Defaults to `None`.
            force_cache_sync (bool, optional): If True, forces synchronization of the dataset cache.
                Defaults to `False`.
            download_videos (bool, optional): Whether to download associated videos.
                Defaults to `True`.
            video_backend (str | None, optional): Backend to use for video decoding.
                Defaults to `None`.
            batch_encoding_size (int, optional): Number of samples per encoded batch.
                Defaults to `1`.

        Raises:
            ImportError: If `lerobot` is not installed.
        """
        super().__init__()

        if LeRobotDataset is None:
            msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
            raise ImportError(msg)

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
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._lerobot_dataset)

    def __getitem__(self, idx: int) -> Observation:
        """Get an item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            Observation: The item from the dataset.
        """
        return FormatConverter.to_observation(self._lerobot_dataset[idx])

    @staticmethod
    def from_lerobot(lerobot_dataset: LeRobotDataset) -> _LeRobotDatasetAdapter:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.

        Returns:
            _LeRobotDatasetAdapter: A new adapter instance that uses the provided dataset.
        """
        instance = _LeRobotDatasetAdapter.__new__(_LeRobotDatasetAdapter)
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
    def fps(self) -> int:
        """Frames per second of dataset."""
        return self._lerobot_dataset.fps

    @property
    def tolerance_s(self) -> float:
        """Tolerance to keep delta timestamps in sync with fps."""
        return self._lerobot_dataset.tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Expose delta_indices from the dataset."""
        indices = self._lerobot_dataset.delta_indices
        return indices if indices is not None else {}

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Allow setting delta_indices on the dataset."""
        self._lerobot_dataset.delta_indices = indices


__all__ = ["_LeRobotDatasetAdapter"]

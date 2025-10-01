# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobotDataset standard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from lightning_utilities import module_available

from getiaction.data import DataModule, Dataset, Observation

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import torch

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None


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
    """Convert item from lerobot to our internal Observation format.

    Args:
        lerobot_item (dict): The item from the lerobot dataset.

    Returns:
        Observation: The observation in our internal format.

    Raises:
        ValueError: If the item is missing a required key.
    """
    required_keys = [
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "timestamp",
    ]
    lerobot_item_keys = lerobot_item.keys()
    for key in required_keys:
        if key not in lerobot_item_keys:
            msg = f"Missing required key: {key}. Available keys: {lerobot_item_keys}"
            raise ValueError(msg)

    if not any(k.startswith("observation") for k in lerobot_item_keys):
        msg = f"Sample must contain some form of observation. Sample keys {lerobot_item_keys}"
        raise ValueError(msg)
    if not any(k.startswith("action") for k in lerobot_item_keys):
        msg = f"Sample must contain an action. Sample keys {lerobot_item_keys}"
        raise ValueError(msg)

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
        images=cast("Any", images) if images is not None else {},
        state=cast("Any", state) if state is not None else None,
        action=cast("Any", action) if action is not None else None,
        task=cast("Any", task) if task is not None else None,
        episode_index=lerobot_item["episode_index"],
        frame_index=lerobot_item["frame_index"],
        index=lerobot_item["index"],
        task_index=lerobot_item["task_index"],
        timestamp=lerobot_item["timestamp"],
        extra=extra,
    )


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
        delta_timestamps: dict[list[float], str] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> None:
        """Initialize a _LeRobotDatasetAdapter.

        This adapter initializes an internal `LeRobotDataset` using the provided
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
        return _convert_lerobot_item_to_observation(self._lerobot_dataset[idx])

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


class LeRobotDataModule(DataModule):
    """A PyTorch Lightning DataModule for the integration of LeRobot datasets.

    This DataModule simplifies the process of using datasets from the Hugging Face Hub
    that follow the LeRobot format. It automatically handles downloading, caching,
    and preparing the dataset for use in a `getiaction` training pipeline.

    The module wraps the original `LeRobotDataset` to make it compatible with the
    expected `Observation` data format and `getiaction.data.Dataset` interface.

    Examples:
        >>> # 1. Instantiate from a Hugging Face repository ID
        >>> datamodule_from_repo = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
        ...     train_batch_size=32
        ... )

        >>> # 2. Instantiate from an existing LeRobotDataset object
        >>> from lerobot.datasets import LeRobotDataset
        >>> raw_dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
        >>> datamodule_from_instance = LeRobotDataModule(
        ...     dataset=raw_dataset,
        ...     train_batch_size=32
        ... )
    """

    def __init__(
        self,
        *,
        repo_id: str | None = None,
        dataset: LeRobotDataset | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        train_batch_size: int = 16,
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

        This class seamlessly integrates a `LeRobotDataset` with the `getiaction`
        training pipeline by automatically adapting it to the required format.

        Args:
            train_batch_size (int, optional): Batch size for the training DataLoader. Defaults to 16.
            repo_id (str | None, optional): Repository ID for the LeRobot dataset.
            Required if `dataset` is not provided.
            dataset (LeRobotDataset | None, optional): Pre-initialized LeRobotDataset instance.
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
            **action_datamodule_kwargs: Additional keyword arguments passed to the base `DataModule`.

        Raises:
            ValueError: If neither `repo_id` nor `dataset` is provided.
            TypeError: If `dataset` is not of type `LeRobotDataset`.
            ImportError: If `lerobot` is not installed.
        """
        if dataset is not None and repo_id is not None:
            msg = "Cannot provide both 'repo_id' and 'dataset'. Please provide only one."
            raise ValueError(msg)

        if dataset is not None:
            if LeRobotDataset is None:
                msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
                raise ImportError(msg)
            if not isinstance(dataset, LeRobotDataset):
                msg = f"The provided 'dataset' must be an instance of LeRobotDataset, but got {type(dataset)}."
                raise TypeError(msg)
            adapted_dataset = _LeRobotDatasetAdapter.from_lerobot(dataset)

        elif repo_id is not None:
            # Initialize the LeRobot dataset adapter directly
            adapted_dataset = _LeRobotDatasetAdapter(
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
            msg = "Must provide either 'repo_id' or a 'dataset' instance."
            raise ValueError(msg)

        # Pass the adapted dataset into the parent class
        super().__init__(
            train_dataset=adapted_dataset,
            train_batch_size=train_batch_size,
            **action_datamodule_kwargs,
        )

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot DataModule for PyTorch Lightning integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lightning_utilities import module_available
from torch.utils.data import DataLoader

from getiaction.data import DataModule

from .converters import DataFormat
from .dataset import _LeRobotDatasetAdapter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from getiaction.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None


class LeRobotDataModule(DataModule):
    """A PyTorch Lightning DataModule for the integration of LeRobot datasets.

    This DataModule simplifies the process of using datasets from the Hugging Face Hub
    that follow the LeRobot format. It automatically handles downloading, caching,
    and preparing the dataset for use in a `getiaction` training pipeline.

    By default, the module wraps the LeRobotDataset in the getiaction `Observation` format.
    For LeRobot policies that expect the original dict format, use `data_format="lerobot"`.

    Examples:
        >>> # 1. Use getiaction format (default)
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
        ...     train_batch_size=32
        ... )

        >>> # 2. Use LeRobot's original dict format
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
        ...     train_batch_size=32,
        ...     data_format="lerobot"
        ... )

        >>> # 3. Using enum (type-safe)
        >>> from getiaction.data.lerobot import DataFormat
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_transfer_cube_human",
        ...     train_batch_size=32,
        ...     data_format=DataFormat.LEROBOT
        ... )

        >>> # 4. Instantiate from an existing LeRobotDataset object
        >>> from lerobot.datasets import LeRobotDataset
        >>> raw_dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
        >>> datamodule = LeRobotDataModule(
        ...     dataset=raw_dataset,
        ...     train_batch_size=32,
        ...     data_format="lerobot"
        ... )
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        repo_id: str | None = None,
        dataset: LeRobotDataset | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        train_batch_size: int = 16,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        data_format: Literal["getiaction", "lerobot"] | DataFormat = "getiaction",
        # Base DataModule parameters (val/test gyms)
        val_gym: Gym | None = None,
        num_rollouts_val: int = 10,
        test_gym: Gym | None = None,
        num_rollouts_test: int = 10,
        max_episode_steps: int | None = 300,
    ) -> None:
        """Initialize a LeRobot-specific Action DataModule.

        Args:
            repo_id (str | None, optional): Repository ID for the LeRobot dataset.
                Required if `dataset` is not provided.
                Defaults to `None`.
            dataset (LeRobotDataset | None, optional): Pre-initialized LeRobotDataset instance.
                Defaults to `None`.
            root (str | Path | None, optional): Local directory for caching dataset files.
                Defaults to `None`.
            episodes (list[int] | None, optional): Specific episode indices to include.
                Defaults to `None`.
            train_batch_size (int, optional): Batch size for the training DataLoader.
                Defaults to `16`.
            image_transforms (Callable | None, optional): Transformations to apply to images.
                Defaults to `None`.
            delta_timestamps (dict[str, list[float]] | None, optional): Mapping of signal keys
                to timestamp offsets.
                Defaults to `None`.
            tolerance_s (float, optional): Tolerance in seconds for aligning timestamps.
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
            data_format (Literal["getiaction", "lerobot"] | DataFormat, optional):
                Output format for the data. Use "getiaction" for the native `Observation` format,
                or "lerobot" for LeRobot's original dict format.
                Defaults to "getiaction".
            val_gym (Gym | None, optional): Validation gym environment.
                Defaults to `None`.
            num_rollouts_val (int, optional): Number of rollouts for validation.
                Defaults to 10.
            test_gym (Gym | None, optional): Test gym environment.
                Defaults to `None`.
            num_rollouts_test (int, optional): Number of rollouts for testing.
                Defaults to `10`.
            max_episode_steps (int | None, optional): Maximum steps per episode.
                Defaults to `300`.

        Raises:
            ValueError: If neither `repo_id` nor `dataset` is provided, or if invalid `data_format`.
            TypeError: If `dataset` is not of type `LeRobotDataset`.
            ImportError: If `lerobot` is not installed.
        """
        if dataset is not None and repo_id is not None:
            msg = "Cannot provide both 'repo_id' and 'dataset'. Please provide only one."
            raise ValueError(msg)

        # Convert `data_format` to enum if it's a string
        self.data_format = DataFormat(data_format)

        # Create the appropriate dataset based on format
        if dataset is not None:
            if LeRobotDataset is None:
                msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
                raise ImportError(msg)
            if not isinstance(dataset, LeRobotDataset):
                msg = f"The provided 'dataset' must be an instance of LeRobotDataset, but got {type(dataset)}."
                raise TypeError(msg)

            train_dataset = (
                _LeRobotDatasetAdapter.from_lerobot(dataset) if data_format == DataFormat.GETIACTION else dataset
            )

        elif repo_id is not None:
            if data_format == DataFormat.GETIACTION:
                train_dataset = _LeRobotDatasetAdapter(
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
                if LeRobotDataset is None:
                    msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
                    raise ImportError(msg)

                train_dataset = LeRobotDataset(
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

        # Pass the dataset to the parent class
        super().__init__(
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            val_gym=val_gym,
            num_rollouts_val=num_rollouts_val,
            test_gym=test_gym,
            num_rollouts_test=num_rollouts_test,
            max_episode_steps=max_episode_steps,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for training.

        Returns data in the format specified by `data_format`:
        - "getiaction": Returns `Observation` dataclass instances (uses custom collate)
        - "lerobot": Returns dict instances in LeRobot's native format (uses default collate)

        Returns:
            DataLoader: Training DataLoader with specified format.
        """
        # For getiaction format, use parent's implementation which has the custom collate function
        if self.data_format == DataFormat.GETIACTION:
            return super().train_dataloader()

        # For lerobot format, use default PyTorch collate to preserve dict structure
        return DataLoader(
            self.train_dataset,
            num_workers=4,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )


__all__ = ["LeRobotDataModule"]

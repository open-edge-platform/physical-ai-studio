# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot DataModule for PyTorch Lightning integration."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from huggingface_hub import hf_hub_download
from lerobot.utils.constants import HF_LEROBOT_HOME
from lightning_utilities import module_available
from torch.utils.data import DataLoader

from physicalai.data import DataModule

from .converters import DataFormat
from .dataset import _LeRobotDatasetAdapter
from .utils.quantile_stats import augment_dataset_quantile_stats, has_quantile_stats

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.gyms import Gym

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

logger = logging.getLogger(__name__)


def _read_total_episodes(repo_id: str, root: str | Path | None) -> int:
    """Read total_episodes from a LeRobot dataset's meta/info.json.

    Path resolution matches ``LeRobotDataset``:
    - When *root* is provided: ``root / meta / info.json``
    - When *root* is ``None``: uses ``HF_LEROBOT_HOME / repo_id``
      (downloads from HuggingFace Hub if not cached locally).

    Args:
        repo_id: Dataset repository ID or folder name.
        root: Dataset root directory (the folder that contains meta/).

    Returns:
        Total number of episodes in the dataset.

    Raises:
        FileNotFoundError: If the dataset metadata is not found at the expected path.
    """
    base = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    info_path = base / "meta" / "info.json"
    if not info_path.exists():
        if root is not None:
            msg = (
                f"Cannot read dataset metadata: {info_path} not found. "
                f"Ensure the dataset exists at '{base}' or specify 'episodes' explicitly."
            )
            raise FileNotFoundError(msg)

        # HuggingFace dataset not cached — download just the info.json
        msg = f"Downloading dataset metadata from HuggingFace: {repo_id}"
        logger.info(msg)
        hf_hub_download(  # nosec B615 - revision param available for pinning
            repo_id=repo_id,
            repo_type="dataset",
            filename="meta/info.json",
            local_dir=base,
        )

    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)

    return int(info["total_episodes"])


class LeRobotDataModule(DataModule):
    """A PyTorch Lightning DataModule for the integration of LeRobot datasets.

    This DataModule simplifies the process of using datasets from the Hugging Face Hub
    that follow the LeRobot format. It automatically handles downloading, caching,
    and preparing the dataset for use in a `physicalai` training pipeline.

    By default, the module wraps the LeRobotDataset in the physicalai `Observation` format.
    For LeRobot policies that expect the original dict format, use `data_format="lerobot"`.

    Examples:
        >>> # 1. Use physicalai format (default)
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
        >>> from physicalai.data.lerobot import DataFormat
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

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        *,
        repo_id: str | None = None,
        dataset: LeRobotDataset | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        train_batch_size: int = 16,
        num_workers: int | Literal["auto"] = "auto",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        data_format: Literal["physicalai", "lerobot"] | DataFormat = "physicalai",
        # Eval-loss validation
        val_split: float = 0.0,
        val_split_seed: int | None = None,
        val_batch_size: int = 1,
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
            num_workers (int | Literal["auto"], optional): Number of DataLoader workers.
                ``"auto"`` (default) uses ``min(cpu_count, 8)``.
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
            data_format (Literal["physicalai", "lerobot"] | DataFormat, optional):
                Output format for the data. Use "physicalai" for the native `Observation` format,
                or "lerobot" for LeRobot's original dict format.
                Defaults to "physicalai".
            val_split (float, optional): Fraction of episodes to hold out for eval-loss
                validation (e.g. ``0.1`` for 10%). The last N episodes are used as the
                validation set. Must be in ``[0, 1)``. ``0`` disables eval-loss validation.
                Defaults to ``0.0``.
            val_split_seed (int | None, optional): Random seed for the train/val episode split.
                ``None`` (default) uses the global ``random`` module, which respects
                ``seed_everything()``. Set an explicit int to use an isolated RNG
                independent of the global seed. Defaults to ``None``.
            val_batch_size (int, optional): Batch size for the eval-loss validation DataLoader.
                Defaults to ``1``.
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
        # Auto-derive repo_id from root directory name when not provided
        if repo_id is None and root is not None:
            repo_id = Path(root).name

        if dataset is not None and repo_id is not None:
            msg = "Cannot provide both 'repo_id' and 'dataset'. Please provide only one."
            raise ValueError(msg)

        if not 0.0 <= val_split < 1.0:
            msg = f"'val_split' must be in [0, 1), got {val_split}."
            raise ValueError(msg)

        if val_split > 0 and dataset is not None:
            msg = (
                "Cannot use 'val_split' with a pre-initialized 'dataset'. "
                "Use 'repo_id' instead (for local datasets, combine 'repo_id' with 'root')."
            )
            raise ValueError(msg)

        if val_split > 0 and val_gym is not None:
            msg = "Cannot use both 'val_split' and 'val_gym'. Choose eval-loss or gym-based validation."
            raise ValueError(msg)

        # Convert `data_format` to enum if it's a string
        self.data_format = DataFormat(data_format)

        # Split episodes into train / val based on val_split
        train_episodes = episodes
        val_episodes: list[int] | None = None
        if val_split > 0 and repo_id is not None:
            total_episodes = _read_total_episodes(repo_id, root)
            all_episodes = episodes if episodes is not None else list(range(total_episodes))
            n_val = max(1, int(len(all_episodes) * val_split))
            # Use isolated RNG if seed given, otherwise global random (respects seed_everything)
            if val_split_seed is not None:
                rng = random.Random(val_split_seed)  # noqa: S311
                val_episodes = sorted(rng.sample(all_episodes, n_val))
            else:
                val_episodes = sorted(random.sample(all_episodes, n_val))
            train_episodes = sorted(ep for ep in all_episodes if ep not in set(val_episodes))
            logger.warning(
                "Val split (%.0f%%): %d val episodes %s, %d train episodes (of %d total)",
                val_split * 100,
                len(val_episodes),
                val_episodes,
                len(train_episodes),
                len(all_episodes),
            )

        # Create the appropriate dataset based on format
        val_eval_dataset = None
        if dataset is not None:
            if LeRobotDataset is None:
                msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
                raise ImportError(msg)
            if not isinstance(dataset, LeRobotDataset):
                msg = f"The provided 'dataset' must be an instance of LeRobotDataset, but got {type(dataset)}."
                raise TypeError(msg)

            train_dataset = (
                _LeRobotDatasetAdapter.from_lerobot(dataset) if data_format == DataFormat.PHYSICALAI else dataset
            )

        elif repo_id is not None:
            if data_format == DataFormat.PHYSICALAI:
                train_dataset = _LeRobotDatasetAdapter(
                    repo_id=repo_id,
                    root=root,
                    episodes=train_episodes,
                    image_transforms=image_transforms,
                    delta_timestamps=delta_timestamps,
                    tolerance_s=tolerance_s,
                    revision=revision,
                    force_cache_sync=force_cache_sync,
                    download_videos=download_videos,
                    video_backend=video_backend,
                    batch_encoding_size=batch_encoding_size,
                )
                if val_episodes is not None:
                    val_eval_dataset = _LeRobotDatasetAdapter(
                        repo_id=repo_id,
                        root=root,
                        episodes=val_episodes,
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
                    episodes=train_episodes,
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

        # Ensure quantile stats (q01/q99) exist — older datasets only have
        # mean/std/min/max.  Compute over all episodes (pre-split) so the
        # normalizer sees the full data distribution.
        if data_format == DataFormat.PHYSICALAI:
            if repo_id is not None:
                self._ensure_quantile_stats(
                    train_dataset,
                    val_eval_dataset,
                    repo_id=repo_id,
                    root=root,
                    revision=revision,
                    force_cache_sync=force_cache_sync,
                    download_videos=download_videos,
                    video_backend=video_backend,
                )
            elif isinstance(train_dataset, _LeRobotDatasetAdapter):
                # Pre-built dataset path: compute quantiles on it directly
                lr_ds = train_dataset._lerobot_dataset  # noqa: SLF001
                if not has_quantile_stats(lr_ds):
                    logger.info("Pre-built dataset lacks quantile stats — computing")
                    augment_dataset_quantile_stats(lr_ds)

        # Pass the dataset to the parent class
        super().__init__(
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            num_workers=num_workers,
            val_gym=val_gym,
            num_rollouts_val=num_rollouts_val,
            val_eval_dataset=val_eval_dataset,
            val_batch_size=val_batch_size,
            test_gym=test_gym,
            num_rollouts_test=num_rollouts_test,
            max_episode_steps=max_episode_steps,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for training.

        Returns data in the format specified by `data_format`:
        - "physicalai": Returns `Observation` dataclass instances (uses custom collate)
        - "lerobot": Returns dict instances in LeRobot's native format (uses default collate)

        Returns:
            DataLoader: Training DataLoader with specified format.
        """
        # For physicalai format, use parent's implementation which has the custom collate function
        if self.data_format == DataFormat.PHYSICALAI:
            return super().train_dataloader()

        # For lerobot format, use default PyTorch collate to preserve dict structure
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )

    @staticmethod
    def _ensure_quantile_stats(
        train_dataset: _LeRobotDatasetAdapter,
        val_dataset: _LeRobotDatasetAdapter | None,
        *,
        repo_id: str,
        root: str | Path | None,
        revision: str | None,
        force_cache_sync: bool,
        download_videos: bool,
        video_backend: str | None,
    ) -> None:
        """Compute q01/q99 on the full (unsplit) dataset when missing.

        Creates a temporary ``LeRobotDataset`` with **all** episodes,
        computes quantile stats, then patches them into the train (and
        optionally val) adapter's underlying ``meta.stats``.
        """
        # Quick check using the train adapter's stats
        train_lerobot_ds = train_dataset._lerobot_dataset  # noqa: SLF001
        if has_quantile_stats(train_lerobot_ds):
            return

        logger.info("Dataset lacks quantile stats — computing from all episodes")

        # Build a full-dataset instance (no episode filter)
        full_ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )
        augment_dataset_quantile_stats(full_ds)

        # Propagate computed quantiles to the train adapter
        for key, feat_stats in full_ds.meta.stats.items():
            if "q01" in feat_stats and key in train_lerobot_ds.meta.stats:
                train_lerobot_ds.meta.stats[key]["q01"] = feat_stats["q01"]
                train_lerobot_ds.meta.stats[key]["q99"] = feat_stats["q99"]

        # Propagate to val adapter if it exists
        if val_dataset is not None:
            val_lerobot_ds = val_dataset._lerobot_dataset  # noqa: SLF001
            for key, feat_stats in full_ds.meta.stats.items():
                if "q01" in feat_stats and key in val_lerobot_ds.meta.stats:
                    val_lerobot_ds.meta.stats[key]["q01"] = feat_stats["q01"]
                    val_lerobot_ds.meta.stats[key]["q99"] = feat_stats["q99"]


__all__ = ["LeRobotDataModule"]

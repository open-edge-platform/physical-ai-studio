# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot dataset adapter for PhysicalAI compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning_utilities import module_available

from physicalai.data.dataset import Dataset
from physicalai.data.observation import Feature, FeatureType, NormalizationParameters, Observation

from .converters import FormatConverter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from physicalai.data import Observation

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
else:
    LeRobotDataset = None

# Number of dimensions of a single image feature (height, width, channel).
_IMAGE_NDIM = 3


def _concat_columns(item: dict, columns: list[str], target_key: str, drop_prefix: str) -> dict:
    """Return a copy of `item` with `columns` concatenated (last dim) into a single `target_key` tensor.

    All keys starting with `drop_prefix` (the original split sub-columns) are removed, so downstream
    conversion sees exactly one combined column instead of a dict of sub-columns.

    Raises:
        KeyError: If any requested column is missing from `item`.
    """
    missing = [col for col in columns if col not in item]
    if missing:
        msg = f"Cannot combine into '{target_key}': missing column(s) {missing}. Available keys: {sorted(item)}"
        raise KeyError(msg)
    tensors = [torch.as_tensor(item[col]) for col in columns]
    max_dim = max(t.dim() for t in tensors)
    expanded = []
    for t in tensors:
        while t.dim() < max_dim:
            t = t.unsqueeze(-1)
        expanded.append(t)
    combined = torch.cat(expanded, dim=-1)
    result = {key: value for key, value in item.items() if not key.startswith(drop_prefix)}
    result[target_key] = combined
    return result


def _combined_feature(
    columns: list[str],
    dataset_meta: LeRobotDatasetMetadata,
    ftype: FeatureType,
    name: str,
) -> Feature:
    """Return a single `Feature` describing the concatenation of `columns`.

    Per-column normalization stats are concatenated in the same order as `columns`; ``q01``/``q99``
    are only kept when present on every column. The combined shape is the sum of the (1-D) column widths.
    """
    stat_keys = ("mean", "std", "min", "max", "q01", "q99")
    accum: dict[str, list[float]] = {key: [] for key in stat_keys}
    present: dict[str, bool] = dict.fromkeys(stat_keys, True)
    total = 0
    for col in columns:
        stats = dataset_meta.stats[col]
        total += int(dataset_meta.features[col]["shape"][0])
        for stat in stat_keys:
            if stat in stats:
                accum[stat].extend(stats[stat].tolist())
            else:
                present[stat] = False
    return Feature(
        ftype=ftype,
        normalization_data=NormalizationParameters(
            mean=accum["mean"] if present["mean"] else None,
            std=accum["std"] if present["std"] else None,
            min=accum["min"] if present["min"] else None,
            max=accum["max"] if present["max"] else None,
            q01=accum["q01"] if present["q01"] else None,
            q99=accum["q99"] if present["q99"] else None,
        ),
        shape=(total,),
        name=name,
    )


class _LeRobotDatasetAdapter(Dataset):
    """An internal adapter that makes a `LeRobotDataset` compatible with the `physicalai.data.Dataset` interface.

    This adapter class serves two primary purposes:
    1.  **Protocol Compliance**: It wraps the `LeRobotDataset` to ensure it conforms to the
        abstract methods and properties required by the `physicalai.data.Dataset` base class
        (e.g., providing `.features`, `.fps`, etc.).
    2.  **Interface Adaptation**: It transforms the dictionary-based output of `LeRobotDataset.__getitem__`
        into the structured `Observation` dataclass format expected by the training pipeline.

    Note:
        This is an internal implementation detail and is not meant to be used directly by end-users.
        The `LeRobotDataModule` handles the creation and management of this adapter automatically.
    """

    _state_columns: list[str] | None = None
    _action_columns: list[str] | None = None

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
        state_columns: list[str] | None = None,
        action_columns: list[str] | None = None,
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
            state_columns (list[str] | None, optional): Ordered LeRobot sub-column keys to concatenate
                into a single ``observation.state`` tensor (e.g. DROID's
                ``["observation.state.joint_positions", "observation.state.gripper_position"]``). All
                ``observation.state.*`` sub-columns are dropped once combined. Defaults to `None` (no combining).
            action_columns (list[str] | None, optional): Ordered LeRobot sub-column keys to concatenate
                into a single ``action`` tensor (e.g. DROID's
                ``["action.joint_position", "action.gripper_position"]``). All ``action.*`` sub-columns are
                dropped once combined. Defaults to `None` (no combining).

        Raises:
            ImportError: If `lerobot` is not installed.
        """
        super().__init__()

        if LeRobotDataset is None:
            msg = "LeRobotDataset is not available. Install lerobot with: uv pip install lerobot."
            raise ImportError(msg)

        self._state_columns = state_columns
        self._action_columns = action_columns

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
        item = self._lerobot_dataset[idx]
        item = self._combine_split_columns(item)
        return FormatConverter.to_observation(item)

    def _combine_split_columns(self, item: dict) -> dict:
        """Return `item` with any configured split state/action sub-columns concatenated into one column."""
        if self._state_columns:
            item = _concat_columns(item, self._state_columns, "observation.state", "observation.state.")
        if self._action_columns:
            item = _concat_columns(item, self._action_columns, "action", "action.")
        return item

    @staticmethod
    def from_lerobot(
        lerobot_dataset: LeRobotDataset,
        *,
        state_columns: list[str] | None = None,
        action_columns: list[str] | None = None,
    ) -> _LeRobotDatasetAdapter:
        """Creates an instance of LeRobotActionDataset from an existing LeRobotDataset instance.

        This static method is useful when you already have a `LeRobotDataset` object
        that you want to wrap for use in action training.

        Args:
            lerobot_dataset (LeRobotDataset): The existing LeRobotDataset instance to be wrapped.
            state_columns (list[str] | None, optional): Ordered LeRobot sub-column keys to concatenate
                into a single ``observation.state`` tensor. Defaults to `None` (no combining).
            action_columns (list[str] | None, optional): Ordered LeRobot sub-column keys to concatenate
                into a single ``action`` tensor. Defaults to `None` (no combining).

        Returns:
            _LeRobotDatasetAdapter: A new adapter instance that uses the provided dataset.
        """
        instance = _LeRobotDatasetAdapter.__new__(_LeRobotDatasetAdapter)
        # Bypassing __init__ to set the internal dataset
        instance._lerobot_dataset = lerobot_dataset  # noqa: SLF001
        instance._state_columns = state_columns  # noqa: SLF001
        instance._action_columns = action_columns  # noqa: SLF001
        return instance

    @property
    def raw_features(self) -> dict[str, dict[Any, Any]]:
        """Raw dataset features."""
        return self._lerobot_dataset.features

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Observation features from the dataset."""
        dataset_features = self._lerobot_dataset.features
        raw_obs_features = {key: ft for key, ft in dataset_features.items() if key.startswith("observation")}
        dataset_meta = self._lerobot_dataset.meta
        state_columns = set(self._state_columns or [])

        observation_features = {}
        for k in raw_obs_features:
            if k in state_columns:
                continue  # folded into the combined "state" feature below
            if k in dataset_meta.features:
                feature_name = k[len("observation.") :]  # Remove "observation." prefix, filtering was done above
                feature_type = FeatureType.STATE
                feature_shape = dataset_meta.features[k]["shape"]
                if dataset_meta.features[k]["dtype"] in {"image", "video"}:
                    feature_type = FeatureType.VISUAL
                    # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0
                    # for ported datasets.
                    if "images." in feature_name:
                        feature_name = feature_name[len("images.") :]
                    elif feature_name.startswith("image."):
                        # Some datasets (e.g. nvidia DROID) use a singular "observation.image.<cam>" prefix.
                        feature_name = feature_name[len("image.") :]
                    names = dataset_meta.features[k].get("names")
                    if names is not None and len(names) >= _IMAGE_NDIM and names[2] in {"channel", "channels"}:
                        feature_shape = (feature_shape[2], feature_shape[0], feature_shape[1])  # (h, w, c) -> (c, h, w)
                    elif names is None and len(feature_shape) == _IMAGE_NDIM and feature_shape[2] in {1, 3, 4}:
                        # v3.0 video features carry names=None; infer channel-last layout from the shape.
                        feature_shape = (feature_shape[2], feature_shape[0], feature_shape[1])
                elif k == "observation.environment_state":
                    feature_type = FeatureType.ENV

                stats = dataset_meta.stats[k]
                observation_features[feature_name] = Feature(
                    ftype=feature_type,
                    normalization_data=NormalizationParameters(
                        mean=stats["mean"].tolist(),
                        std=stats["std"].tolist(),
                        min=stats["min"].tolist(),
                        max=stats["max"].tolist(),
                        q01=stats["q01"].tolist() if "q01" in stats else None,
                        q99=stats["q99"].tolist() if "q99" in stats else None,
                    ),
                    shape=feature_shape,
                    name=feature_name,
                )

        if self._state_columns:
            observation_features["state"] = _combined_feature(
                self._state_columns,
                dataset_meta,
                FeatureType.STATE,
                "state",
            )
        return observation_features

    @property
    def action_features(self) -> dict[str, Feature]:
        """Action features from LeRobot dataset."""
        dataset_features = self._lerobot_dataset.features
        raw_act_features = {key: ft for key, ft in dataset_features.items() if key.startswith("action")}
        dataset_meta = self._lerobot_dataset.meta
        action_columns = set(self._action_columns or [])

        action_features = {}
        for k in raw_act_features:
            if k in action_columns:
                continue  # folded into the combined "action" feature below
            if k in dataset_meta.features:
                stats = dataset_meta.stats[k]
                action_features[k] = Feature(
                    ftype=FeatureType.ACTION,
                    normalization_data=NormalizationParameters(
                        mean=stats["mean"].tolist(),
                        std=stats["std"].tolist(),
                        min=stats["min"].tolist(),
                        max=stats["max"].tolist(),
                        q01=stats["q01"].tolist() if "q01" in stats else None,
                        q99=stats["q99"].tolist() if "q99" in stats else None,
                    ),
                    shape=dataset_meta.features[k]["shape"],
                    name=k,
                )

        if self._action_columns:
            action_features["action"] = _combined_feature(
                self._action_columns,
                dataset_meta,
                FeatureType.ACTION,
                "action",
            )
        return action_features

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
        """Expose delta_indices from the DatasetReader (lerobot >=0.5.1)."""
        reader = getattr(self._lerobot_dataset, "reader", None)
        if reader is not None:
            return reader.delta_indices or {}
        return getattr(self._lerobot_dataset, "delta_indices", None) or {}

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        """Set delta_indices on DatasetReader so the training callback can inject them post-construction."""
        reader = getattr(self._lerobot_dataset, "reader", None)
        if reader is not None:
            reader.delta_indices = indices
        else:
            self._lerobot_dataset.delta_indices = indices


__all__ = ["_LeRobotDatasetAdapter"]

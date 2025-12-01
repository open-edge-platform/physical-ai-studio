# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types and internal representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Observation:
    """A single observation or batch of observations from an imitation learning dataset.

    This dataclass can represent both:
    - A single sample (unbatched): tensors have shape [feature_dim]
    - A batch of samples (batched): tensors have shape [batch_size, feature_dim]

    Provides convenient methods for format conversion:
    - `to_dict()`: Convert to nested dictionary
    - `from_dict()`: Create Observation from dictionary

    Supports dict-like interface for iteration and access:
    - `keys()`: Get all field names
    - `values()`: Get all field values
    - `items()`: Get (field_name, value) tuples

    For framework-specific conversions (e.g., LeRobot format), use the appropriate
    converter from `getiaction.data.lerobot.converters`.

    Examples:
        >>> # Single observation
        >>> obs = Observation(
        ...     action=torch.tensor([1.0, 2.0]),
        ...     images={"top": torch.rand(3, 224, 224)}
        ... )

        >>> # Batch of observations (from collate_fn)
        >>> batch = Observation(
        ...     action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # [B, action_dim]
        ...     images={"top": torch.rand(8, 3, 224, 224)}  # [B, C, H, W]
        ... )

        >>> # Convert for use with LeRobot policies
        >>> from getiaction.data.lerobot import FormatConverter
        >>> lerobot_dict = FormatConverter.to_lerobot_dict(batch)
    """

    # Core Observation
    action: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    task: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    state: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    images: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None

    # Optional RL & Metadata Fields
    next_reward: torch.Tensor | np.ndarray | None = None
    next_success: bool | None = None
    episode_index: torch.Tensor | np.ndarray | None = None
    frame_index: torch.Tensor | np.ndarray | None = None
    index: torch.Tensor | np.ndarray | None = None
    task_index: torch.Tensor | np.ndarray | None = None
    timestamp: torch.Tensor | np.ndarray | None = None
    info: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None

    class FieldName(StrEnum):
        """Observation field name constants for dict access and type annotations."""

        ACTION = "action"
        TASK = "task"
        STATE = "state"
        IMAGES = "images"

        NEXT_REWARD = "next_reward"
        NEXT_SUCCESS = "next_success"
        EPISODE_INDEX = "episode_index"
        FRAME_INDEX = "frame_index"
        INDEX = "index"
        TASK_INDEX = "task_index"
        TIMESTAMP = "timestamp"
        INFO = "info"
        EXTRA = "extra"

    def to_dict(self, *, flatten: bool = True) -> dict[str, Any]:
        """Convert Observation to a dictionary format.

        Returns a dictionary with the same structure as the Observation fields,
        preserving nested dictionaries (e.g., images with multiple cameras) if flatten is False.
        Otherwise, flattens nested dictionaries into keys with dot notation.

        Returns:
            dict[str, Any]: Dictionary representation with optional nested structure.

        Examples:
            >>> obs = Observation(action=torch.tensor([1.0, 2.0]))
            >>> d = obs.to_dict()
            >>> # d = {"action": tensor([1.0, 2.0]), "task": None, ...}
        """
        if not flatten:
            return asdict(self)
        flat_dict = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                key_entries = []
                for sub_key, sub_value in value.items():
                    flat_dict[f"{key}.{sub_key}"] = sub_value
                    key_entries.append(f"{key}.{sub_key}")
                flat_dict[f"_{key}_keys"] = key_entries
            else:
                flat_dict[key] = value

        return flat_dict

    @staticmethod
    def get_flattened_keys(data: dict[str, Any], field: Observation.FieldName | str) -> list[str]:
        """Retrieve all keys associated with a specific component from the data dictionary.

        This method checks for component keys in the following ways:
        1. Directly if the component exists as a key in the data dictionary
        2. Through a cached list of keys stored with the pattern "_{component}_keys"
        3. As a fallback, by searching for keys that start with "{component}."
        Args:
            data: Dictionary containing observation data and component keys
            field: The field identifier to search for, either as an
                      Observation.FieldName enum value or a string

        Returns:
            A list of string keys associated with the component. Returns a list
            containing the component itself if it exists directly in data, the
            cached list of keys if available, or an empty list if neither exists.

        Example:
            >>> data = {"label": {...}, "_label_keys": ["label1", "label2"]}
            >>> Observation.get_flattened_keys(data, "label")
            ["label"]
            >>> Observation.get_flattened_keys(data, "annotation")
            ["label1", "label2"]
        """
        if field in data:
            return [field]

        if f"_{field}_keys" in data:
            return data[f"_{field}_keys"]

        return [key for key in data if key.startswith(f"{field}.")]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Create an Observation from a dictionary.

        Args:
            data: Dictionary with observation fields.

        Returns:
            Observation: New Observation instance.

        Examples:
            >>> data = {"action": torch.tensor([1.0, 2.0]), "state": torch.tensor([0.5])}
            >>> obs = Observation.from_dict(data)
        """
        # Filter to only known fields
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    @classmethod
    def keys(cls) -> list[str]:
        """Return list of all possible observation field names.

        Returns:
            list[str]: List of all field names defined in the dataclass.

        Examples:
            >>> # Get all possible field names
            >>> Observation.keys()
            ['action', 'task', 'state', 'images', 'next_reward', ...]

            >>> # Works on instance too
            >>> obs = Observation(action=torch.tensor([1.0]))
            >>> obs.keys()
            ['action', 'task', 'state', 'images', 'next_reward', ...]
        """
        return [f.name for f in fields(cls)]

    def values(self) -> list[Any]:
        """Return list of all field values (including None).

        Returns:
            list[Any]: List of all field values in the same order as keys().

        Examples:
            >>> obs = Observation(action=torch.tensor([1.0]), state=torch.tensor([2.0]))
            >>> values = obs.values()
            >>> # [tensor([1.0]), None, tensor([2.0]), None, ...]
        """
        return [getattr(self, f.name) for f in fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        """Return list of (field_name, value) tuples.

        Returns:
            list[tuple[str, Any]]: List of (key, value) pairs.

        Examples:
            >>> obs = Observation(action=torch.tensor([1.0]), state=torch.tensor([2.0]))
            >>> for key, value in obs.items():
            ...     if value is not None:
            ...         print(f"{key}: {value}")
            action: tensor([1.0])
            state: tensor([2.0])
        """
        return [(f.name, getattr(self, f.name)) for f in fields(self)]

    def to(self, device: str | torch.device) -> Observation:
        """Move torch tensors to specified device.

        This method moves all torch.Tensor fields to the target device (e.g., CPU, CUDA).
        NumPy arrays and other field types are left unchanged. Works recursively with
        nested dictionaries (e.g., multi-camera images).

        Args:
            device: Target device for torch tensors (e.g., "cpu", "cuda", "cuda:0").
                Accepts anything that torch.Tensor.to() accepts.

        Returns:
            Observation: New Observation instance with tensors moved to target device.

        Examples:
            >>> obs = Observation(
            ...     action=torch.tensor([1.0, 2.0]),
            ...     images={"top": torch.rand(3, 224, 224)}
            ... )

            >>> # Move to CUDA
            >>> obs_cuda = obs.to(device="cuda")
            >>> obs_cuda = obs.to("cuda")  # equivalent

            >>> # Move back to CPU
            >>> obs_cpu = obs_cuda.to(device="cpu")

            >>> # Works with batched observations
            >>> batch = Observation(action=torch.randn(8, 2))
            >>> batch_gpu = batch.to("cuda")
        """

        def _move_to_device(value: dict | torch.Tensor | np.ndarray | None) -> dict | torch.Tensor | np.ndarray | None:
            """Recursively move torch tensors to device.

            Returns:
                Value with torch tensors moved to the specified device.
            """
            if isinstance(value, dict):
                return {k: _move_to_device(v) for k, v in value.items()}
            if isinstance(value, torch.Tensor):
                return value.to(device)
            # Non-tensor types (None, bool, numpy arrays, etc.) pass through
            return value

        # Create new instance with all fields moved
        new_dict = {k: _move_to_device(v) for k, v in self.items()}
        return Observation.from_dict(new_dict)

    def to_numpy(self) -> Observation:
        """Convert all torch tensors to numpy arrays.

        This method creates a new Observation instance with all torch.Tensor fields
        converted to numpy arrays. Works recursively with nested dictionaries (e.g.,
        multi-camera images). Non-tensor fields and existing numpy arrays pass through unchanged.

        Useful for inference pipelines where models expect numpy inputs (e.g., ONNX, OpenVINO).

        Returns:
            Observation: New Observation instance with numpy arrays instead of torch tensors.

        Examples:
            >>> obs = Observation(
            ...     action=torch.tensor([1.0, 2.0]),
            ...     state=torch.tensor([0.5]),
            ...     images={"top": torch.rand(3, 224, 224)}
            ... )
            >>> obs_np = obs.to_numpy()
            >>> isinstance(obs_np.action, np.ndarray)  # True
            >>> isinstance(obs_np.images["top"], np.ndarray)  # True

        See Also:
            :meth:`to` - Move tensors to different devices (CPU/GPU)
            :meth:`to_torch` - Convert numpy arrays to torch tensors
        """

        def _to_numpy(value: dict | torch.Tensor | np.ndarray | None) -> dict | np.ndarray | None:
            """Recursively convert torch tensors to numpy arrays.

            Returns:
                Value with torch tensors converted to numpy arrays.
            """
            if isinstance(value, dict):
                return {k: _to_numpy(v) for k, v in value.items()}
            if isinstance(value, torch.Tensor):
                return value.cpu().numpy()
            # Non-tensor types (None, bool, numpy arrays, etc.) pass through
            return value

        # Create new instance with all fields converted
        new_dict = {k: _to_numpy(v) for k, v in self.items()}
        return Observation.from_dict(new_dict)

    def to_torch(self, device: str | torch.device = "cpu") -> Observation:
        """Convert all numpy arrays to torch tensors on specified device.

        This method creates a new Observation instance with all numpy array fields
        converted to torch tensors. Works recursively with nested dictionaries (e.g.,
        multi-camera images). Non-array fields and existing torch tensors pass through unchanged.

        Useful for converting inference inputs back to torch format for training or
        when transitioning between numpy-based and torch-based pipelines.

        Args:
            device: Target device for created tensors (default: "cpu").
                Accepts anything that torch.Tensor.to() accepts.

        Returns:
            Observation: New Observation instance with torch tensors instead of numpy arrays.

        Examples:
            >>> import numpy as np
            >>> obs = Observation(
            ...     action=np.array([1.0, 2.0]),
            ...     state=np.array([0.5]),
            ...     images={"top": np.random.rand(3, 224, 224)}
            ... )
            >>> obs_torch = obs.to_torch()
            >>> isinstance(obs_torch.action, torch.Tensor)  # True
            >>> isinstance(obs_torch.images["top"], torch.Tensor)  # True

            >>> # Create tensors directly on GPU
            >>> obs_cuda = obs.to_torch(device="cuda")

        See Also:
            :meth:`to` - Move tensors to different devices (CPU/GPU)
            :meth:`to_numpy` - Convert torch tensors to numpy arrays
        """

        def _to_torch(value: dict | torch.Tensor | np.ndarray | None) -> dict | torch.Tensor | None:
            """Recursively convert numpy arrays to torch tensors.

            Returns:
                Value with numpy arrays converted to torch tensors.
            """
            if isinstance(value, dict):
                return {k: _to_torch(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).to(device)
            # Non-array types (None, bool, torch tensors, etc.) pass through
            return value

        # Create new instance with all fields converted
        new_dict = {k: _to_torch(v) for k, v in self.items()}
        return Observation.from_dict(new_dict)

    def __getitem__(self, idx: int | slice) -> Observation:
        """Index into batched observation to extract sample(s).

        Supports both integer indexing (single sample) and slice indexing (sub-batch).
        Works recursively with nested dictionaries (e.g., multi-camera images).

        Args:
            idx: Integer index or slice to extract. Use slice(0, 1) or 0:1 to
                preserve batch dimension when extracting first sample.

        Returns:
            Observation: New Observation with indexed data. Preserves structure
                and field types (torch.Tensor or np.ndarray).

        Examples:
            >>> # Batch of 8 observations
            >>> batch = Observation(
            ...     action=torch.randn(8, 2),
            ...     images={"top": torch.rand(8, 3, 224, 224)}
            ... )

            >>> # Get first sample (squeeze batch dimension)
            >>> sample = batch[0]
            >>> sample.action.shape  # torch.Size([2])

            >>> # Get first sample (preserve batch dimension)
            >>> sample = batch[0:1]
            >>> sample.action.shape  # torch.Size([1, 2])

            >>> # Get sub-batch
            >>> sub_batch = batch[0:4]
            >>> sub_batch.action.shape  # torch.Size([4, 2])
        """

        def _index(value: dict | torch.Tensor | np.ndarray | None) -> dict | torch.Tensor | np.ndarray | None:
            """Recursively index into value.

            Args:
                value: Value to index into.

            Returns:
                Indexed value.
            """
            if value is None:
                return None
            if isinstance(value, dict):
                return {k: _index(v) for k, v in value.items()}
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value[idx]
            # Non-indexable types (bool, scalars, etc.) pass through
            return value

        # Create new instance with all fields indexed
        new_dict = {k: _index(v) for k, v in self.items()}
        return Observation.from_dict(new_dict)


class FeatureType(StrEnum):
    """Enum for feature types."""

    VISUAL = "VISUAL"
    ACTION = "ACTION"
    STATE = "STATE"
    ENV = "ENV"


@dataclass(frozen=True)
class Feature:
    """A feature representation."""

    normalization_data: NormalizationParameters | None = None
    ftype: FeatureType | None = None
    shape: tuple[int, ...] | None = None
    name: str | None = None


@dataclass(frozen=True)
class NormalizationParameters:
    """Parameters for normalizing a tensor."""

    mean: list[float] | float | None = None
    std: list[float] | float | None = None
    min: list[float] | float | None = None
    max: list[float] | float | None = None


# Module-level constants for convenient dict access
# Generated from Observation.FieldName enum to avoid duplication.
#
# Usage: from getiaction.data.observation import STATE, ACTION, IMAGES
# Then: batch[STATE] instead of batch["state"]
#
# Note: All of the following are equivalent for dict access:
# - batch[ACTION]                       (recommended: imported constant)
# - batch["action"]                     (string literal)
# - batch[Observation.FieldName.ACTION] (enum member)
#
# Using imported constants is recommended for IDE autocomplete, refactoring support, and consistency.
ACTION = Observation.FieldName.ACTION.value
EPISODE_INDEX = Observation.FieldName.EPISODE_INDEX.value
EXTRA = Observation.FieldName.EXTRA.value
FRAME_INDEX = Observation.FieldName.FRAME_INDEX.value
IMAGES = Observation.FieldName.IMAGES.value
INDEX = Observation.FieldName.INDEX.value
INFO = Observation.FieldName.INFO.value
NEXT_REWARD = Observation.FieldName.NEXT_REWARD.value
NEXT_SUCCESS = Observation.FieldName.NEXT_SUCCESS.value
STATE = Observation.FieldName.STATE.value
TASK = Observation.FieldName.TASK.value
TASK_INDEX = Observation.FieldName.TASK_INDEX.value
TIMESTAMP = Observation.FieldName.TIMESTAMP.value

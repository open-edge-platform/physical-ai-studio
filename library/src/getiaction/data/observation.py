# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types and internal representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import numpy as np


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

    class ComponentKeys(StrEnum):
        """Enum for batch observation components."""

        STATE = "state"
        ACTION = "action"
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
    def get_flattened_keys(data: dict[str, Any], component: Observation.ComponentKeys | str) -> list[str]:
        """Retrieve all keys associated with a specific component from the data dictionary.

        This method checks for component keys in two ways:
        1. Directly if the component exists as a key in the data dictionary
        2. Through a cached list of keys stored with the pattern "_{component}_keys"
        Args:
            data: Dictionary containing observation data and component keys
            component: The component identifier to search for, either as an
                      Observation.ComponentKeys enum value or a string

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
        if component in data:
            return [component]

        if f"_{component}_keys" in data:
            return data[f"_{component}_keys"]

        return []

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
        """Move all tensors in the observation to the specified device.

        This method creates a new Observation instance with all tensor fields moved
        to the specified device. Works with both single tensors and nested dictionaries
        of tensors. Non-tensor fields are copied as-is.

        Args:
            device: Target device (e.g., "cpu", "cuda", "cuda:0", torch.device("cuda"))

        Returns:
            Observation: New Observation instance with tensors on the target device.

        Examples:
            >>> obs = Observation(
            ...     action=torch.tensor([1.0, 2.0]),
            ...     images={"top": torch.rand(3, 224, 224)}
            ... )
            >>> obs_cuda = obs.to("cuda")  # Move to GPU
            >>> obs_cpu = obs_cuda.to("cpu")  # Move back to CPU

            >>> # Works with batched observations too
            >>> batch = Observation(action=torch.randn(8, 2))
            >>> batch_gpu = batch.to("cuda")
        """

        def _move_to_device(
            value: torch.Tensor | np.ndarray | dict | bool | None,  # noqa: FBT001
        ) -> torch.Tensor | np.ndarray | dict | bool | None:
            """Recursively move tensors to device.

            Returns:
                The value moved to device if it's a tensor, otherwise unchanged.
            """
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: _move_to_device(v) for k, v in value.items()}
            # For non-tensor types (None, bool, numpy arrays, etc.), return as-is
            return value

        # Create new instance with all fields moved to device
        new_dict = {k: _move_to_device(v) for k, v in self.items()}
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

    mean: torch.Tensor | np.ndarray | None = None
    std: torch.Tensor | np.ndarray | None = None
    min: torch.Tensor | np.ndarray | None = None
    max: torch.Tensor | np.ndarray | None = None

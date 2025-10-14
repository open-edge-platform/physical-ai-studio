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


def gym_observation_to_observation(
    gym_obs: dict[str, Any],
    camera_keys: list[str] | None = None,
) -> Observation:
    """Convert raw gym environment observation to Observation dataclass.

    Handles the conversion from gym's raw observation format (numpy arrays, HWC images)
    to our standardized Observation format (torch tensors, CHW images, batched).

    This is used during rollouts to normalize gym environment outputs before passing
    them to policy.select_action().

    Args:
        gym_obs: Raw observation dict from env.step() or env.reset().
            Common keys: "pixels" (image), "agent_pos" (state), etc.
        camera_keys: List of camera names to use for image mapping.
            If None, uses ["top"] as default for single camera.

    Returns:
        Observation with normalized format:
        - Images: CHW format, torch.float32, with batch dimension
          - Single camera: direct tensor (compatible with LeRobot's "observation.image")
          - Multiple cameras: dict of tensors (creates "observation.images.{camera}")
        - State: torch.float32 with batch dimension
        - All on CPU (device transfer happens in select_action)

    Examples:
        >>> # Single camera - creates direct tensor
        >>> gym_obs = {
        ...     "pixels": np.array([[[0.5, 0.3, 0.8]]]),  # HWC numpy
        ...     "agent_pos": np.array([0.5, 0.3])         # 1D numpy
        ... }
        >>> obs = gym_observation_to_observation(gym_obs)
        >>> # obs.images = tensor([1, 3, 1, 1])  # Direct tensor (not dict)
        >>> # obs.state = tensor([1, 2])  # Batched

        >>> # Multiple cameras - creates dict
        >>> obs = gym_observation_to_observation(gym_obs, camera_keys=["top", "wrist"])
        >>> # obs.images = {"top": tensor([1, 3, 1, 1]), "wrist": ...}

        >>> # Also works via class method
        >>> obs = Observation.from_gym(gym_obs)
    """
    if camera_keys is None:
        camera_keys = ["top"]

    images: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | None = None
    state: torch.Tensor | None = None

    # Handle image observations
    if "pixels" in gym_obs:
        pixels = gym_obs["pixels"]

        # Convert to torch tensor
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.from_numpy(pixels)

        # Ensure float32 (gym often returns float64 or uint8)
        if pixels.dtype not in {torch.float32, torch.float16}:
            pixels = pixels.float()

        # Convert HWC → CHW if needed
        if pixels.ndim == 3 and pixels.shape[-1] in {1, 3, 4}:  # noqa: PLR2004
            pixels = pixels.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Add batch dimension
        if pixels.ndim == 3:  # noqa: PLR2004
            pixels = pixels.unsqueeze(0)  # (C, H, W) → (1, C, H, W)

        # For single camera: direct tensor for compatibility with existing models
        # For multiple cameras: dict with camera names
        images = pixels if len(camera_keys) == 1 else {camera_keys[0]: pixels}

    # Handle state observations
    state_keys = ["agent_pos", "state"]  # Common gym state keys
    for key in state_keys:
        if key in gym_obs:
            state_data = gym_obs[key]

            # Convert to torch tensor
            if not isinstance(state_data, torch.Tensor):
                state_data = torch.from_numpy(state_data)

            # Ensure float32
            if state_data.dtype != torch.float32:
                state_data = state_data.float()

            # Add batch dimension
            if state_data.ndim == 1:
                state_data = state_data.unsqueeze(0)  # (D,) → (1, D)

            state = state_data
            break

    return Observation(images=images, state=state)


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

    def to_dict(self) -> dict[str, Any]:
        """Convert Observation to a nested dictionary format.

        Returns a dictionary with the same structure as the Observation fields,
        preserving nested dictionaries (e.g., images with multiple cameras).

        Returns:
            dict[str, Any]: Dictionary representation with nested structure.

        Examples:
            >>> obs = Observation(action=torch.tensor([1.0, 2.0]))
            >>> d = obs.to_dict()
            >>> # d = {"action": tensor([1.0, 2.0]), "task": None, ...}
        """
        return asdict(self)

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

    @classmethod
    def from_gym(
        cls,
        gym_obs: dict[str, Any],
        camera_keys: list[str] | None = None,
    ) -> Observation:
        """Convert raw gym environment observation to Observation.

        Convenience class method that delegates to gym_observation_to_observation().
        Provides discoverable API following the existing from_dict() pattern.

        Args:
            gym_obs: Raw observation dict from env.step() or env.reset()
            camera_keys: List of camera names for image mapping

        Returns:
            Observation instance with normalized format

        Examples:
            >>> gym_obs = {"pixels": np.array([[[0.5]]]), "agent_pos": np.array([0.5, 0.3])}
            >>> obs = Observation.from_gym(gym_obs)
            >>> # obs.images = {"top": tensor([1, 1, 1, 1])}  # Batched, CHW
            >>> # obs.state = tensor([1, 2])  # Batched
        """
        return gym_observation_to_observation(gym_obs, camera_keys)


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

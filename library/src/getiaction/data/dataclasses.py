# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types and internal representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

    For framework-specific conversions (e.g., LeRobot format), use the appropriate
    converter from `getiaction.data.lerobot.converters`.

    Examples:
        >>> # Single observation
        >>> obs = Observation(
        ...     action=torch.tensor([1.0, 2.0]),
        ...     images={"top": torch.rand(3, 224, 224)}
        ... )
        >>>
        >>> # Batch of observations (from collate_fn)
        >>> batch = Observation(
        ...     action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # [B, action_dim]
        ...     images={"top": torch.rand(8, 3, 224, 224)}  # [B, C, H, W]
        ... )
        >>>
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

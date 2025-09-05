# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Types and internal representations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

import torch

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class TensorField:
    """Represents a generic, immutable tensor with convenient metadata.

    Attributes:
        data (torch.Tensor): The underlying tensor data.
        semantic (str | None): Optional semantic description of the tensor's
            contents (e.g., 'positions', 'velocities').
    """

    data: torch.Tensor
    # Optional metadata
    semantic: str | None = None

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Self:
        """Creates a TensorField instance from a NumPy array.

        Args:
            array (np.ndarray): The NumPy array to convert.

        Returns:
            A new TensorField instance.
        """
        return cls(data=torch.from_numpy(array))

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the underlying tensor."""
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Returns the data type of the underlying tensor."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Returns the device of the underlying tensor."""
        return self.data.device

    # NOTE: this creates a copy, not an inplace change,
    # could be a problem if TensorField is large.
    def to(self, device: torch.device | str) -> Self:
        """Moves the tensor data to the specified device.

        Args:
            device (torch.device | str): The target device to move the data to.

        Returns:
            A new TensorField instance with the data on the new device. If the
            tensor is already on the target device, returns the same instance.
        """
        if self.device == device:
            return self
        return replace(self, data=self.data.to(device))

    def to_numpy(self) -> np.ndarray:
        """Converts the tensor to a NumPy array.

        Note: The tensor is detached andmoved to the CPU before conversion,
        as NumPy arrays are CPU-based.

        Returns:
            A NumPy ndarray representation of the tensor's data.
        """
        # A tensor must be on the CPU to be converted to numpy.
        return self.data.detach().cpu().numpy()


@dataclass(frozen=True)
class ImageField(TensorField):
    """Represents an image as a TensorField with format metadata.

    Attributes:
        format (str): The format of the image data, e.g., 'CHW' (Channels,
            Height, Width) or 'HWC' (Height, Width, Channels).
    """

    format: str = "CHW"


@dataclass
class Observation:
    """A container for a single environment observation."""

    images: list[ImageField] | None
    state: TensorField | None
    action: TensorField | None
    task: str | None


# TODO: Do we need a lerobot specific observation? extra info may be needed for some policies?
@dataclass
class LeRobotObservation(Observation):
    """An Observation from lerobot dataset"""

    next_reward: TensorField
    next_success: bool
    episode_index: TensorField
    frame_index: TensorField
    index: TensorField
    task_index: TensorField
    timestamp: TensorField


# TODO: How much of Obstype from gymnasium should we support. This changes our implementation
# https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.ObservationWrapper
@dataclass
class GymObservation:
    """An Observation from gymnasium environment"""

    observation: Observation
    reward: TensorField
    termination: bool
    truncation: bool
    info: dict

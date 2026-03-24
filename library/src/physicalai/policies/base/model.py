# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base torch nn.Module for Models."""

from abc import ABC, abstractmethod

from torch import nn


class Model(nn.Module, ABC):
    """Base class for Models.

    Model is an entity that is fully compatible with torch.nn.Module,
    and is used to define the architecture of the neural network inside Policy.
    """

    def __init__(self) -> None:
        """Initialize nn.Module."""
        super().__init__()

    @property
    @abstractmethod
    def reward_delta_indices(self) -> list | None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None or a list of reward indices.
        """

    @property
    @abstractmethod
    def action_delta_indices(self) -> list | None:
        """Get indices of actions relative to the current timestep.

        Returns:
            None or a list of relative action indices.
        """

    @property
    @abstractmethod
    def observation_delta_indices(self) -> list | None:
        """Get indices of observations relative to the current timestep.

        Returns:
            None or a list of relative observation indices.
        """

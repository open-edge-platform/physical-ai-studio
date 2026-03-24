# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base torch nn.Module for Models."""

from abc import ABC, abstractmethod
from typing import Any

from torch import nn


class Model(nn.Module, ABC):
    """Base class for Models.

    Model is an entity that is fully compatible with torch.nn.Module,
    and is used to define the architecture of the neural network inside Policy.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def reward_delta_indices(self) -> Any:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None or a list of reward indices.
        """

    @property
    @abstractmethod
    def action_delta_indices(self) -> Any:
        """Get indices of actions relative to the current timestep.

        Returns:
            None or a list of relative action indices.
        """

    @property
    @abstractmethod
    def observation_delta_indices(self) -> Any:
        """Get indices of observations relative to the current timestep.

        Returns:
            None or a list of relative observation indices.
        """

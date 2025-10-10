# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod

import lightning as L  # noqa: N812
import torch
from torch import nn

from getiaction.data import Observation


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies."""

    def __init__(self) -> None:
        """Initialize the Base Lightning Module for Policies."""
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module

    def forward(self, batch: Observation, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        """Perform forward pass of the policy.

        The input batch is an Observation dataclass that can be converted to
        the format expected by the model using `.to_dict()` or `.to_lerobot_dict()`.

        Args:
            batch (Observation): Input batch of observations
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Model predictions
        """
        del args, kwargs
        return self.model(batch)

    @abstractmethod
    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (Observation): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """

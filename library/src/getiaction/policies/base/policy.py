# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod

import lightning as L  # noqa: N812
import torch
from torch import nn


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies."""

    def __init__(self) -> None:
        """Initialize the Base Lightning Module for Policies."""
        super().__init__()
        self.model: nn.Module

    def forward(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        """Perform forward pass of the policy.

        The input batched is preprocessed before being passed to the model.

        Args:
            batch (dict[str, torch.Tensor]): Input batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Model predictions
        """
        del args, kwargs
        return self.model(batch)

    @abstractmethod
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (dict[str, torch.Tensor]): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """

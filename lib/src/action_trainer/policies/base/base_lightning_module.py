# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies"""

from abc import ABC, abstractmethod
from typing import Dict

import lightning as L
import torch
import torch.nn as nn


class TrainerModule(L.LightningModule, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module

    def forward(self, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
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
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """

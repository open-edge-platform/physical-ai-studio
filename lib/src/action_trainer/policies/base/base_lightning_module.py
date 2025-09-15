# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies"""

from abc import ABC

import lightning as L
import torch
from torch import nn


class ActionTrainerModule(L.LightningModule, ABC):
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

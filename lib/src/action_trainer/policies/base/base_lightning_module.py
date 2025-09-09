# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies"""


import lightning as pl
from abc import ABC, abstractmethod
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch import Callback
from torch import nn
from tensordict import TensorDict


class ActionTrainerModule(pl.LightningModule, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: nn.Module

    def forward(self, batch: TensorDict, *args, **kwargs) -> TensorDict:
        """Perform forward pass of the policy.
        The input batched is preprocessed before being passed to the model.

        Args:
            batch (TensorDict): Input batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            TensorDict: Processed batch with model predictions
        """
        del args, kwargs
        return self.model(batch)

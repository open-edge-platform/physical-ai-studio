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
        self.loss: nn.Module
        self.callbacks: list[Callback]

        self._input_image_sizes: list[tuple[int, int]] | None = None
        self._input_state_shape: tuple[int] | None = None

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
        processed_batch = self._preprocess_observation(batch)
        return self.model(processed_batch)

    @abstractmethod
    def _preprocess_observation(batch: TensorDict) -> TensorDict:
        """Preprocess the input observation batch before passing it to the torch model."""
        raise NotImplementedError

    def test_step(self, batch: TensorDict, batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Perform test step.

        This method is called during the test stage of training. It calls
        the model's forward method to ensure consistency with exported model behavior,
        then merges the predictions into the batch for post-processing.

        Args:
            batch (TensorDict): Input batch
            batch_idx (int): Index of the batch
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Updated batch with model predictions
        """
        del args, kwargs, batch_idx  # These variables are not used.
        processed_batch = self._preprocess_observation(batch)
        predictions = self.model(processed_batch)
        return batch.update(**predictions._asdict())

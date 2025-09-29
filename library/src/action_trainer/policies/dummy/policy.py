# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage"""

from collections.abc import Iterable

import torch

from action_trainer.policies.base import TrainerModule
from action_trainer.policies.dummy.config import DummyConfig
from action_trainer.policies.dummy.model import Dummy as DummyModel


class Dummy(TrainerModule):
    """Dummy policy wrapper."""

    def __init__(self, config: DummyConfig) -> None:
        """Initialize the Dummy policy wrapper.

        This class wraps a `DummyModel` and integrates it into a `TrainerModule`,
        validating the action shape and preparing the model for training.

        Args:
            config (DummyConfig): Configuration object containing the action shape
                and other hyperparameters required for initializing the policy.

        Raises:
            ValueError: If the `action_shape` in the configuration is None.
            TypeError: If the `action_shape` is not a valid type (e.g., string or non-iterable).
        """
        super().__init__()
        self.config = config
        self.action_shape = self._validate_action_shape(self.config.action_shape)

        # model
        self.model = DummyModel(self.action_shape)

    def _validate_action_shape(self, shape: torch.Size | Iterable) -> torch.Size:
        """Validate and normalize the action shape.

        Args:
            shape (torch.Size | Iterable): The input shape to validate.

        Returns:
            torch.Size: A validated torch.Size object.

        Raises:
            ValueError: If `shape` is `None`.
            TypeError: If `shape` is not a valid type (e.g., string).
        """
        if shape is None:
            msg = "Action is missing a 'shape' key in its features dictionary."
            raise ValueError(msg)

        if isinstance(shape, torch.Size):
            return shape

        if isinstance(shape, str):
            msg = f"Shape for action '{shape}' must be a sequence of numbers, but received a string."
            raise TypeError(msg)

        if isinstance(shape, Iterable):
            return torch.Size(shape)

        msg = f"The 'action_shape' argument must be a torch.Size or Iterable, but received type {type(shape).__name__}."
        raise TypeError(msg)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        return self.model.select_action(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Dict[str, torch.Tensor]): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        del batch_idx  # Unused variable
        loss, loss_dict = self.forward(batch)  # noqa: RUF059
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the policy.

        Returns:
            torch.optim.Optimizer: Adam optimizer over the model parameters.
        """
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def evaluation_step(self, batch: dict[str, torch.Tensor], stage: str) -> None:
        """Evaluation step (no-op by default).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage  # Unused variables

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step (calls evaluation_step).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        del batch_idx  # Unused variable
        return self.evaluation_step(batch=batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step (calls evaluation_step).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        del batch_idx  # Unused variable
        return self.evaluation_step(batch=batch, stage="test")

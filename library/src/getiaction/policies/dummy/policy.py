# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage."""

import torch
from torch import nn

from getiaction.policies.base import Policy
from getiaction.policies.dummy.config import OptimizerConfig


class Dummy(Policy):
    """Dummy policy wrapper."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Initialize the Dummy policy wrapper.

        This class wraps a model and integrates it into a Lightning module.

        Args:
            model (nn.Module): The model to use for the policy.
            optimizer (torch.optim.Optimizer | None): Optimizer to use for the policy. If `None`,
                a default `Adam` optimizer will be created with the model parameters and a learning rate of 1e-4.
                Defaults to `None`.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    @staticmethod
    def _create_optimizer(config: OptimizerConfig, model: nn.Module) -> torch.optim.Optimizer:
        """Create an optimizer from configuration.

        Args:
            config (OptimizerConfig): Optimizer configuration.
            model (nn.Module): Model to create optimizer for.

        Returns:
            torch.optim.Optimizer: Created optimizer.

        Raises:
            ValueError: If optimizer type is not supported.
        """
        if config.optimizer_type.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )
        if config.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        msg = f"Unsupported optimizer type: {config.optimizer_type}"
        raise ValueError(msg)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the policy.

        Returns:
            torch.optim.Optimizer: Adam optimizer over the model parameters.
        """
        if self.optimizer is not None:
            return self.optimizer
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

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

    @staticmethod
    def evaluation_step(batch: dict[str, torch.Tensor], stage: str) -> None:
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

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        return self.model.select_action(batch)

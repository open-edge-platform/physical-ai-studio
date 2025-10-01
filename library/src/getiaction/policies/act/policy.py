# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch

from getiaction.policies.act.model import ACT as ACTModel
from getiaction.policies.base.base_lightning_module import TrainerModule


class ACT(TrainerModule):
    def __init__(
        self,
        model: ACTModel,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        super().__init__()

        self.model = model

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

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
        return self.optimizer

    def evaluation_step(self, batch: dict[str, torch.Tensor], stage: str) -> None:
        """Evaluation step (no-op by default).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        return

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step (calls evaluation_step).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        return self.evaluation_step(batch=batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step (calls evaluation_step).

        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            batch_idx (int): Index of the batch.
        """
        return self.evaluation_step(batch=batch, stage="test")

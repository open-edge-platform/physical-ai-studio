# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage."""

import torch

from getiaction.data import Observation
from getiaction.export.mixin_export import Export
from getiaction.gyms import Gym
from getiaction.policies.base import Policy
from getiaction.policies.dummy.config import DummyConfig
from getiaction.policies.dummy.model import Dummy as DummyModel
from getiaction.policies.utils import FromCheckpoint


class Dummy(FromCheckpoint, Export, Policy):
    """Dummy policy wrapper."""

    model_type: type = DummyModel
    model_config_type: type = DummyConfig

    def __init__(self, model: DummyModel) -> None:
        """Initialize the Dummy policy wrapper.

        This class wraps a `DummyModel` and integrates it into a `TrainerModule`,
        validating the action shape and preparing the model for training.

        Args:
            model (DummyModel): An instance of the DummyModel class.
        """
        super().__init__()
        self.model: DummyModel = model

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch: Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """
        # Get action from model
        return self.model.select_action(batch.to_dict())  # type: ignore[attr-defined]

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Perform forward pass of the Dummy policy.

        The return value depends on the model's training mode:
        - In training mode: Returns (loss, loss_dict) from the model's forward method
        - In evaluation mode: Returns action predictions via select_action method

        Args:
            batch (Observation): Input batch of observations

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: In training mode, returns
                tuple of (loss, loss_dict). In eval mode, returns selected actions tensor.
        """
        if self.training:
            # During training, return loss information for backpropagation
            return self.model(batch.to_dict())

        # During evaluation, return action predictions
        return self.select_action(batch)

    @staticmethod
    def _generate_example_inputs() -> dict[str, torch.Tensor]:
        """Generate example inputs for export.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with dummy observation inputs.
        """
        return {"state": torch.randn(1, 4)}

    def training_step(self, batch: Observation, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Observation): The training batch.
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the loss.
        """
        del batch_idx
        loss, _ = self.model(batch.to_dict())
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

    @staticmethod
    def evaluation_step(batch: Observation, stage: str) -> None:
        """Evaluation step (no-op by default).

        Args:
            batch (Observation): Input batch.
            stage (str): Evaluation stage, e.g., "val" or "test".
        """
        del batch, stage  # Unused variables

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step.

        Runs gym-based validation via rollout evaluation. The DataModule's val_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step.

        Runs gym-based testing via rollout evaluation. The DataModule's test_dataloader
        returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate.
            batch_idx: Index of the batch.

        Returns:
            Metrics dict from gym rollout.
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def reset(self) -> None:
        """Reset the policy state.

        Dummy policy has no state to reset, so this is a no-op.
        """

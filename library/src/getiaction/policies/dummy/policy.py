# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy lightning module and policy for testing usage."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from getiaction.data import Observation
from getiaction.gyms import Gym
from getiaction.policies.base import Policy
from getiaction.policies.dummy.config import DummyConfig
from getiaction.policies.dummy.model import Dummy as DummyModel


class Dummy(Policy):
    """Dummy policy wrapper."""

    def __init__(self, config: DummyConfig) -> None:
        """Initialize the Dummy policy wrapper.

        This class wraps a `DummyModel` and integrates it into a `TrainerModule`,
        validating the action shape and preparing the model for training.

        Args:
            config (DummyConfig): Configuration object containing the action shape
                and other hyperparameters required for initializing the policy.
        """
        super().__init__()
        self.config = config
        self.action_shape = self._validate_action_shape(self.config.action_shape)

        # model
        self.model = DummyModel(self.action_shape)

    @staticmethod
    def _validate_action_shape(shape: torch.Size | Iterable) -> torch.Size:
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

    def select_action(self, batch: Observation | dict[str, Any]) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch: Input batch - can be Observation (training) or dict (rollout).

        Returns:
            torch.Tensor: Selected actions.
        """
        # Convert numpy to tensors and add batch dim if needed
        batch_dict = {
            k: torch.from_numpy(v).unsqueeze(0).float() if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        # Get action from model
        action = self.model.select_action(batch_dict)  # type: ignore[attr-defined]

        # Remove batch dim if present (rollout expects unbatched)
        return action.squeeze(0) if action.ndim > 1 and action.shape[0] == 1 else action

    def training_step(self, batch: Observation, batch_idx: int) -> dict[str, torch.Tensor]:
        """Training step for the policy.

        Args:
            batch (Observation): The training batch.
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

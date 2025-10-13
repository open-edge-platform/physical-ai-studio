# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod

import lightning as L  # noqa: N812
import torch
from torch import nn

from getiaction.data import Observation
from getiaction.data.observation import GymObservation
from getiaction.eval import rollout


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies."""

    def __init__(self) -> None:
        """Initialize the Base Lightning Module for Policies."""
        super().__init__()
        self.model: nn.Module

    def transfer_batch_to_device(
        self,
        batch: Observation,
        device: torch.device,
        dataloader_idx: int,
    ) -> Observation:
        """Transfer batch to device.

        PyTorch Lightning hook to move custom batch types to the correct device.
        This is called automatically by Lightning before the batch is passed to
        training_step, validation_step, etc.

        For Observation objects, uses the custom .to(device) method.
        For other types, delegates to the parent class implementation.

        Args:
            batch: The batch to move to device
            device: Target device
            dataloader_idx: Index of the dataloader (unused, required by Lightning API)

        Returns:
            Batch moved to the target device
        """
        if isinstance(batch, Observation):
            return batch.to(device)

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def forward(self, batch: Observation, *args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003
        """Perform forward pass of the policy.

        The input batch is an Observation dataclass that can be converted to
        the format expected by the model using `.to_dict()` or `.to_lerobot_dict()`.

        Args:
            batch (Observation): Input batch of observations
            *args: Additional positional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Model predictions
        """
        del args, kwargs
        return self.model(batch.to_dict())

    @abstractmethod
    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch (Observation): Input batch of observations.

        Returns:
            torch.Tensor: Selected actions.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy state.

        This method should be called when the environment is reset. It clears
        internal state such as action queues, observation histories, and any
        other stateful components used by the policy.

        For example:
        - Action chunking policies clear their action queue
        - Diffusion policies reset observation/action deques
        - Recurrent policies reset hidden states

        This is critical for proper evaluation in gym environments, where
        each episode must start with a clean slate.
        """

    def evaluate_gym(self, batch: GymObservation, batch_idx: int, stage: str) -> dict[str, float]:
        """Evaluate policy on gym environment and log metrics.

        This is a helper method used by both validation_step and test_step to avoid
        code duplication. It runs a rollout in the gym environment and logs metrics.

        Args:
            batch: GymObservation containing the environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)
            stage: Either "val" or "test" for metric prefix

        Returns:
            Dictionary of metrics (though metrics are also logged via self.log_dict)
        """
        # Run rollout
        result = rollout(
            env=batch.env,
            policy=self,
            seed=batch.seed if batch.seed is not None else batch_idx,
            max_steps=batch.max_steps,
            return_observations=False,
        )

        # Log metrics with appropriate prefix
        metrics = {
            f"{stage}/gym/episode_length": result["episode_length"],
            f"{stage}/gym/sum_reward": result["sum_reward"],
            f"{stage}/gym/max_reward": result["max_reward"],
            f"{stage}/gym/success": float(result["is_success"]),
        }

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return metrics

    def validation_step(self, batch: GymObservation, batch_idx: int) -> dict[str, float]:
        """Validation step for the policy.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns GymObservation batches containing
        the gym environment to evaluate.

        Args:
            batch: GymObservation containing the environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)

        Returns:
            Dictionary of metrics from the gym rollout evaluation
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: GymObservation, batch_idx: int) -> dict[str, float]:
        """Test step for the policy.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns GymObservation batches containing
        the gym environment to evaluate.

        Args:
            batch: GymObservation containing the environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)

        Returns:
            Dictionary of metrics from the gym rollout evaluation
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

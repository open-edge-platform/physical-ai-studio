# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L  # noqa: N812
import torch
from torch import nn

from getiaction.data import Observation
from getiaction.eval import Rollout
from getiaction.gyms import Gym


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies."""

    def __init__(self) -> None:
        """Initialize the Base Lightning Module for Policies."""
        super().__init__()
        self.model: nn.Module

        # Initialize torchmetrics-based rollout metrics for validation and testing
        self.val_rollout = Rollout()
        self.test_rollout = Rollout()

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

    @abstractmethod
    def forward(self, batch: Observation) -> Any:  # noqa: ANN401
        """Perform forward pass of the policy.

        The behavior of this method depends on the model's training mode:
        - In training mode: Should return loss information for backpropagation
          (typically a loss tensor or tuple of (loss, loss_dict))
        - In evaluation mode: Should return action predictions for environment interaction
          (typically via calling self.select_action(batch))

        The input batch is an Observation dataclass that can be converted to
        the format expected by the model using `.to_dict()` or `.to_lerobot_dict()`.

        Args:
            batch (Observation): Input batch of observations

        Returns:
            The return type depends on the training mode and specific policy implementation:
            - Training mode: Loss information (torch.Tensor or tuple[torch.Tensor, dict])
            - Evaluation mode: Action predictions (torch.Tensor)

        Example implementation:
            ```python
            def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict]:
                if self.training:
                    return self.model(batch)
                return self.select_action(batch)
            ```
        """

    @abstractmethod
    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select an action using the policy model.

        Args:
            batch: Input batch of observations.

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

    def evaluate_gym(self, batch: Gym, batch_idx: int, stage: str) -> dict[str, float]:
        """Evaluate policy on gym environment and log metrics using torchmetrics.

        This method uses the torchmetrics-based Rollout for proper distributed
        synchronization and state management. It runs a rollout and updates the
        appropriate metric (val or test), which will be aggregated at epoch end.

        Args:
            batch: Gym environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)
            stage: Either "val" or "test" for metric prefix

        Returns:
            Dictionary of per-episode metrics with stage prefix (for compatibility)
        """
        # Select the appropriate metric based on stage
        metric = self.val_rollout if stage == "val" else self.test_rollout

        # Update metric with this rollout
        metric.update(env=batch, policy=self, seed=batch_idx)

        # Get the most recent episode metrics from the metric state
        latest_metrics = {
            "sum_reward": metric.all_sum_rewards[-1].item(),  # type: ignore[index]
            "max_reward": metric.all_max_rewards[-1].item(),  # type: ignore[index]
            "episode_length": int(metric.all_episode_lengths[-1].item()),  # type: ignore[index]
            "success": float(metric.all_successes[-1].item()),  # type: ignore[index]
        }

        # Log per-episode metrics (on_step=True for immediate feedback)
        per_episode_dict = {f"{stage}/gym/episode/{k}": v for k, v in latest_metrics.items()}
        self.log_dict(per_episode_dict, on_step=True, on_epoch=False, batch_size=1)

        # Return metrics with prefix (for backward compatibility and Lightning consumption)
        return {f"{stage}/gym/{k}": v for k, v in latest_metrics.items()}

    def validation_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Validation step for the policy.

        Runs gym-based validation by executing rollouts in the environment.
        The DataModule's val_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)

        Returns:
            Dictionary of metrics from the gym rollout evaluation
        """
        return self.evaluate_gym(batch, batch_idx, stage="val")

    def test_step(self, batch: Gym, batch_idx: int) -> dict[str, float]:
        """Test step for the policy.

        Runs gym-based testing by executing rollouts in the environment.
        The DataModule's test_dataloader returns Gym environment instances directly.

        Args:
            batch: Gym environment to evaluate
            batch_idx: Index of the batch (used as seed for reproducibility)

        Returns:
            Dictionary of metrics from the gym rollout evaluation
        """
        return self.evaluate_gym(batch, batch_idx, stage="test")

    def on_validation_epoch_end(self) -> None:
        """Compute and log aggregated validation metrics at the end of the epoch.

        This hook is called by Lightning after all validation_step calls are complete.
        It computes aggregated statistics across all rollouts and logs them with
        proper distributed synchronization.
        """
        # Compute aggregated metrics (automatically synced across GPUs)
        metrics = self.val_rollout.compute()

        # Log aggregated metrics (exclude n_episodes from logging)
        aggregated_dict = {f"val/gym/{k}": v for k, v in metrics.items() if k != "n_episodes"}
        self.log_dict(aggregated_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        # Reset metric for next epoch
        self.val_rollout.reset()

    def on_test_epoch_end(self) -> None:
        """Compute and log aggregated test metrics at the end of the test run.

        This hook is called by Lightning after all test_step calls are complete.
        It computes aggregated statistics across all rollouts and logs them with
        proper distributed synchronization.
        """
        # Compute aggregated metrics (automatically synced across GPUs)
        metrics = self.test_rollout.compute()

        # Log aggregated metrics (exclude n_episodes from logging)
        aggregated_dict = {f"test/gym/{k}": v for k, v in metrics.items() if k != "n_episodes"}
        self.log_dict(aggregated_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        # Reset metric for next test run
        self.test_rollout.reset()

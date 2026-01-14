# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import lightning as L  # noqa: N812
import torch
from torch import nn

from getiaction.data import Observation
from getiaction.eval import Rollout
from getiaction.gyms import Gym


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies.

    Provides common functionality for all policies including:
    - Action queue management for action chunking
    - Gym evaluation with torchmetrics
    - Device transfer hooks
    """

    def __init__(self, n_action_steps: int = 1) -> None:
        """Initialize the Base Lightning Module for Policies.

        Args:
            n_action_steps: Number of action steps to execute per chunk.
                Used for action queue sizing. Defaults to 1 (no chunking).
        """
        super().__init__()
        # Only set model attribute if the subclass hasn't defined it as a property
        # (e.g., LeRobot wrappers define model as a property that returns _lerobot_policy)
        if not isinstance(getattr(type(self), "model", None), property):
            self.model: nn.Module | None = None

        # Action queue for action chunking (unified across all policies)
        self._action_queue: deque[torch.Tensor] = deque(maxlen=n_action_steps)
        self._n_action_steps = n_action_steps

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
        - In evaluation mode: Should return action chunk predictions

        The input batch is an Observation dataclass that can be converted to
        the format expected by the model using `.to_dict()` or `.to_lerobot_dict()`.

        Args:
            batch (Observation): Input batch of observations

        Returns:
            The return type depends on the training mode and specific policy implementation:
            - Training mode: Loss information (torch.Tensor or tuple[torch.Tensor, dict])
            - Evaluation mode: Action chunk tensor of shape (B, T, D) or (T, D)

        Example implementation:
            ```python
            def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict]:
                if self.training:
                    return self.model(batch)
                return self.predict_action_chunk(batch)
            ```
        """

    @abstractmethod
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions from observation.

        This is the core inference method that predicts multiple future actions
        (action chunking). Subclasses must implement this method.

        Args:
            batch: Input batch of observations.

        Returns:
            Action chunk tensor of shape (B, T, D) or (T, D) where T is chunk size.
        """

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select a single action using action chunking with queue.

        This method implements the standard action chunking pattern:
        1. Check if there are queued actions from previous predictions
        2. If queue is empty, predict a new action chunk via predict_action_chunk()
        3. Queue the predicted actions and return the first one

        For policies that don't use action chunking (n_action_steps=1),
        this simply calls predict_action_chunk() and returns the action.

        Args:
            batch: Input batch of observations.

        Returns:
            Single action tensor of shape (B, D) or (D,).
        """
        # Check queue first
        queued = self._get_queued_action()
        if queued is not None:
            return queued

        # Predict new action chunk and queue
        actions = self.predict_action_chunk(batch)
        return self._queue_actions(actions)

    def reset(self) -> None:
        """Reset the policy state.

        Clears the action queue and any other stateful components.
        Called when the environment is reset to start a new episode.
        """
        self._action_queue.clear()

    def _queue_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Queue predicted actions and return the first one.

        This implements action chunking: the model predicts multiple actions,
        they are queued, and returned one at a time on subsequent calls.

        Args:
            actions: Predicted actions of shape (B, T, D) or (T, D).

        Returns:
            First action from the queue.
        """
        # Handle (B, T, D) -> split along T dimension
        if actions.dim() == 3:  # noqa: PLR2004
            # Transpose to (T, B, D), then extend queue
            self._action_queue.extend(actions.transpose(0, 1))
        else:
            # Already (T, D), just extend
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    def _get_queued_action(self) -> torch.Tensor | None:
        """Get next action from queue if available.

        Returns:
            Next queued action or None if queue is empty.
        """
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()
        return None

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

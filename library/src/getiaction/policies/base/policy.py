# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Lightning Module for Policies."""

from abc import ABC, abstractmethod
from typing import Any

import lightning as L  # noqa: N812
import torch
from torch import nn

from getiaction.data import Observation


class Policy(L.LightningModule, ABC):
    """Base Lightning Module for Policies."""

    def __init__(self) -> None:
        """Initialize the Base Lightning Module for Policies."""
        super().__init__()
        self.save_hyperparameters()

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
        return self.model(batch)

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

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor | None:
        """Validation step for the policy.

        This handles two types of validation:
        1. Dataset validation: batch contains observations and actions
        2. Gym validation: batch contains a gym environment under 'env' key

        For gym validation, the evaluation is handled by GymEvaluation
        which calls the rollout() function. This method just needs to return
        None to signal that the callback should handle it.

        Args:
            batch: Either an Observation batch or dict with 'env' key
            batch_idx: Index of the batch

        Returns:
            Loss tensor for dataset validation, None for gym validation
        """
        del batch_idx  # Unused argument

        # Check if this is a gym validation batch
        if isinstance(batch, dict) and "env" in batch:
            # Gym validation is handled by GymEvaluation
            return None

        # Standard dataset validation - subclasses can override for custom logic
        msg = "validation_step must be implemented for dataset validation"
        raise NotImplementedError(msg)

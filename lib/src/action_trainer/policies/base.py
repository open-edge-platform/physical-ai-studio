# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base class for policies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from action_trainer.data.types import Observation, TensorField


class ActionPolicy(nn.Module, ABC):
    """
    An abstract base class for imitation learning policies.

    This class defines the standard interface for policies, separating the concerns of
    training (forward pass for loss computation) and inference (action selection).
    Any concrete policy must implement all three abstract methods.
    """

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Computes the model's output for a batch, typically for loss calculation.

        This method is called during the training loop. It should take a batch of
        data (including ground truth actions) and return a dictionary containing
        at least the predicted actions and the computed loss.

        Args:
            batch (dict[str, Any]): A dictionary of batched data from the DataLoader.
                e.g., {
                    'images': torch.Tensor of shape [B, C, H, W],
                    'state': torch.Tensor of shape [B, state_dim],
                    'action': torch.Tensor of shape [B, action_dim], # Ground truth
                }

        Returns:
            A dictionary containing tensors for loss computation.
            e.g., {
                'action_pred': predicted_actions_tensor,
                'loss': computed_loss_tensor
            }
        """

    @abstractmethod
    @torch.inference_mode()
    def predict_action(self, batch: dict[str, Any]) -> TensorField:
        """
        Predicts a chunk of actions from a batch of observations.

        This method is for efficient, batched inference. It should not compute
        gradients. It takes a batch of observations (without ground truth actions)
        and returns the model's predicted actions.

        Args:
            batch (dict[str, Any]): A dictionary of batched observation data.
                e.g., {
                    'images': torch.Tensor of shape [B, C, H, W],
                    'state': torch.Tensor of shape [B, state_dim],
                }

        Returns:
            A TensorField containing the batch of predicted actions.

        Example Usage:
            # Create a batch of 16 dummy states
            state_tensor = torch.randn(16, 10)
            observation_batch = {'state': state_tensor}

            # Get a batch of predicted actions
            predicted_actions = policy.predict_action(observation_batch)
            # --> predicted_actions.shape might be (16, 4)
        """

    @abstractmethod
    @torch.inference_mode()
    def select_action(self, observation: Observation) -> TensorField:
        """
        Selects a single action given a single, unbatched observation.

        This is the primary method used for inference in a live environment loop.
        Implementations should handle converting the single `Observation` object
        into a batch of size 1, running inference, and returning a single action.

        Args:
            observation (Observation): A single observation object from the environment.

        Returns:
            A TensorField containing the single predicted action.

        Example Usage in an Environment Loop:
            obs, info = env.reset()  # `obs` is a single Observation object
            done = False
            while not done:
                action_field = policy.select_action(obs)
                action_np = action_field.to_numpy()
                obs, reward, done, truncated, info = env.step(action_np)
        """

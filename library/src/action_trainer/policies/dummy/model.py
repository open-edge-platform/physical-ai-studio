# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy for testing usage."""

from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _infer_batch_size(batch: dict[str, Any]) -> int:
    """Infer the batch size from the first tensor in the batch.

    This function scans the values of the input batch dictionary and returns
    the size of the first dimension of the first `torch.Tensor` it finds. It
    assumes that all tensors in the batch have the same batch dimension.

    Args:
        batch (dict[str, Any]): A dictionary where values may include tensors.

    Returns:
        int: The inferred batch size.

    Raises:
        ValueError: If no tensor is found in the batch.
    """
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return v.shape[0]
    msg = "Could not infer batch size from batch."
    raise ValueError(msg)


class Dummy(nn.Module):
    """A dummy model for testing training and evaluation loops.

    This model simulates behavior of an action-predicting model by returning
    random actions and optionally managing temporal ensembles or action queues.
    """

    def __init__(
        self,
        action_shape: torch.Size,
        n_action_steps: int = 1,
        temporal_ensemble_coeff: float | None = None,
        n_obs_steps: int = 1,
        horizon: int | None = None,
    ) -> None:
        """Initialize the DummyModel.

        Args:
            action_shape (torch.Size): The shape of a single action.
            n_action_steps (int, optional): Number of action steps per chunk.
                Defaults to 1.
            temporal_ensemble_coeff (float | None, optional): Coefficient for
                temporal ensembling. If `None`, an action queue is used instead.
                Defaults to `None`.
            n_obs_steps (int, optional): Number of observation steps.
                Defaults to 1.
            horizon (int | None, optional): Prediction horizon. If `None`,
                defaults to `n_action_steps`.
        """
        super().__init__()
        self.action_shape = action_shape
        self.n_action_steps = n_action_steps
        self.temporal_ensemble_coeff = temporal_ensemble_coeff

        # default horizon = number of action steps
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon if horizon is not None else n_action_steps

        if self.temporal_ensemble_coeff is not None:
            # simple placeholder for temporal ensemble
            self.temporal_buffer: None = None
        else:
            self._action_queue: deque = deque(maxlen=self.n_action_steps)

        # dummy parameter for optimizer and backward
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @property
    def observation_delta_indices(self) -> list[int]:
        """Get indices of observations relative to the current timestep.

        Returns:
            list[int]: A list of relative observation indices.
        """
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        """Get indices of actions relative to the current timestep.

        Returns:
            list[int]: A list of relative action indices.
        """
        return list(range(0 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None
        """
        return None

    def reset(self) -> None:
        """Reset internal buffers.

        Clears the temporal buffer (if using temporal ensemble) or the
        action queue (otherwise).
        """
        if self.temporal_ensemble_coeff is not None:
            self.temporal_buffer = None
        else:
            self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action from the model.

        If temporal ensembling is enabled, returns the mean over predicted
        actions. Otherwise, actions are queued and returned sequentially.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            torch.Tensor: A tensor representing the selected action.
        """
        self.eval()

        if self.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            # simple stand-in for ensembler: just take mean
            return actions.mean(dim=1)

        # Handle action queue logic
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of random actions.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            torch.Tensor: A tensor of shape
                `(batch_size, n_action_steps, *action_shape)`.
        """
        batch_size = _infer_batch_size(batch)
        return torch.randn((batch_size, self.n_action_steps, *self.action_shape))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Forward pass through the model.

        If in training mode, returns an MSE loss between predictions and
        zeros. If in evaluation mode, returns a chunk of random actions.

        Args:
            batch (dict[str, torch.Tensor]): A batch of input observations.

        Returns:
            tuple[torch.Tensor, dict]:
                - If training: a scalar loss tensor and dictionary of loss.
                - If evaluating: predicted actions tensor.
        """
        if self.training:
            batch_size = _infer_batch_size(batch)
            # pred now depends on a parameter so it has grad_fn
            pred = (
                torch.randn((batch_size, self.n_action_steps, *self.action_shape), device=self.dummy_param.device)
                + self.dummy_param
            )
            target = torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
            return loss, {"loss_mse": loss}
        return self.predict_action_chunk(batch)

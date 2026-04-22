# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base torch nn.Module for Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class Model(nn.Module, ABC):
    """Base class for Models.

    Model is an entity that is fully compatible with torch.nn.Module,
    and is used to define the architecture of the neural network inside Policy.

    Subclasses must implement:

    - ``forward(batch)``: standard PyTorch forward pass.  In training mode it
      should return ``(loss, loss_dict)``; in eval mode it should return
      predicted actions.
    - ``compute_loss(batch)``: compute the **training** loss (with gradients).
      Called by ``forward()`` when ``self.training`` is ``True``.
    - ``compute_val_loss(batch)``: compute the **validation** loss (no
      gradients).  Override this to use a more meaningful metric than the
      training loss (e.g. action prediction MSE for diffusion / flow-matching
      models).  The default falls back to ``compute_loss``.
    """

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss for this model.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of (loss tensor with grad, dict with at least a ``"loss"`` key).
        """

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the validation loss for this model.

        Override in subclasses to use a different metric from the training
        loss (e.g. action prediction MSE after full denoising).  The default
        delegates to :meth:`compute_loss`.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Tuple of (loss tensor, dict with at least a ``"loss"`` key).
        """
        return self.compute_loss(batch)

    @property
    @abstractmethod
    def reward_delta_indices(self) -> list | None:
        """Return reward indices.

        Currently returns `None` as rewards are not implemented.

        Returns:
            None or a list of reward indices.
        """

    @property
    @abstractmethod
    def action_delta_indices(self) -> list | None:
        """Get indices of actions relative to the current timestep.

        Returns:
            None or a list of relative action indices.
        """

    @property
    @abstractmethod
    def observation_delta_indices(self) -> list | None:
        """Get indices of observations relative to the current timestep.

        Returns:
            None or a list of relative observation indices.
        """

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base class for policies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    import numpy as np

    from action_trainer.data.dataclasses import Observation


class ActionPolicy(nn.Module, ABC):
    """
    An abstract base class for imitation learning policies.
    """

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Computes the model's output for a batch, typically for loss calculation.

        Returns:
            A dictionary containing tensors for loss computation.
            e.g., {'action_pred': ..., 'loss': ...}
        """
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def predict_action(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Predicts a chunk of actions from a batch of observations.

        Returns:
            A dictionary containing the batch of predicted actions.
            e.g., {'action': predicted_actions_tensor}
        """
        raise NotImplementedError

    @abstractmethod
    @torch.inference_mode()
    def select_action(self, observation: Observation) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Selects a single action given a single, unbatched observation.

        Returns:
            A dictionary containing the single predicted action.
            e.g., {'action': single_action_array}
        """
        raise NotImplementedError

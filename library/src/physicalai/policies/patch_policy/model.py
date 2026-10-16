# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Gaoyue Zhou
# Authors: Gaoyue Zhou, Zichen Jeff Cui
# SPDX-License-Identifier: MIT

"""Patch Policy Model."""

from typing import Any

import torch
from torch import Tensor

from physicalai.policies.base import Model

from .config import PatchPolicyConfig


class PatchPolicyModel(Model):
    """Patch Policy Model class."""

    def __init__(
        self,
        input_features: list,
        output_features: list,
        n_action_steps: int = 50,
        chunk_size: int = 50,
        n_obs_steps: int = 1,
    ) -> None:
        """Initialize Patch Policy Model.

        Args:
            input_features: A list of input feature names.
            output_features: A list of output feature names.
            n_action_steps: The number of action steps.
            chunk_size: The chunk size for processing.
            n_obs_steps: The number of observation steps.
        """
        super().__init__()
        self.config = PatchPolicyConfig(
            input_features=input_features,
            output_features=output_features,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            chunk_size=chunk_size,
        )

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute the loss for a given batch.

        Args:
            batch: A dictionary containing the input data and targets.

        Returns:
            A tuple of (loss, metrics), where loss is a scalar tensor and metrics is a dictionary of additional information.
        """
        return torch.tensor(0.0), {}

    def predict_action_chunk(self, observation: Any) -> Tensor:
        """Predict a ``[B, n_action_steps, A]`` action chunk.

        The model emits the full chunk for the latest observation timestep and trims it to the
        configured number of executed steps.
        """
        return torch.randn(
            observation.images[list(observation.images.keys())[0]].shape[0],
            self.config.n_action_steps,
            self.config.action_dim,
        )

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

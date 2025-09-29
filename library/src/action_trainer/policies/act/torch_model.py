# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn


class ACTModel(nn.Module):
    """ACT Model"""

    def __init__(self):
        """Initialize the ACT Model."""
        super().__init__()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select action."""

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict action chunk."""

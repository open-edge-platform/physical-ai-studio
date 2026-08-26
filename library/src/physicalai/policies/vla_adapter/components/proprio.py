# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Derived from https://github.com/OpenHelix-Team/VLA-Adapter
# (prismatic/models/projectors.py), Copyright (c) OpenHelix Team,
# licensed under the MIT License.

"""Projection of proprioceptive state into the action head's embedding space.

*Proprioception* is the robot's own sense of its body configuration — for LIBERO
an 8-vector: end-effector position (3), axis-angle orientation (3), gripper (2).
Cameras do not substitute for it: actions are *relative*, so a correct delta
needs the current pose, and a wrist camera cannot see its own gripper aperture
while the third-person view is routinely occluded.

In Studio terms this is just ``Observation.state`` (``FeatureType.STATE``), so
it normalises through the usual path. Head-side counterpart to
``PrismaticVisualProjector``, which feeds the backbone.
"""

from __future__ import annotations

import torch
from torch import nn


class ProprioProjector(nn.Module):
    """Two-layer MLP lifting proprioceptive state to the backbone width.

    Projecting rather than concatenating lets the state participate as a
    first-class *token* in the head's cross-attention.
    """

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        """Initialize the projector.

        Args:
            llm_dim: Backbone hidden width to project into.
            proprio_dim: Width of the raw proprioceptive vector.
        """
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """Project a proprioceptive state vector.

        Args:
            proprio: State tensor ``(B, proprio_dim)``.

        Returns:
            ``(B, llm_dim)``.
        """
        projected = self.fc1(proprio)
        projected = self.act_fn1(projected)
        return self.fc2(projected)

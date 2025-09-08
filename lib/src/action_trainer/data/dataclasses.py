# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Types and internal representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class Observation:
    """A single observation from an imitation learning dataset."""

    # Core Observation
    action: torch.Tensor | np.ndarray | None
    task: str | None
    state: torch.Tensor | np.ndarray | None = None
    images: dict[str, torch.Tensor | np.ndarray] | None = None

    # Optional RL & Metadata Fields
    next_reward: torch.Tensor | np.ndarray | None = None
    next_success: bool | None = None
    episode_index: torch.Tensor | np.ndarray | None = None
    frame_index: torch.Tensor | np.ndarray | None = None
    index: torch.Tensor | np.ndarray | None = None
    task_index: torch.Tensor | np.ndarray | None = None
    timestamp: torch.Tensor | np.ndarray | None = None
    info: dict[str, Any] | None = None

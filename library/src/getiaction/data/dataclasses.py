# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types and internal representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from enum import Enum

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True)
class Observation:
    """A single observation from an imitation learning dataset."""

    # Core Observation
    action: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    task: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    state: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None
    images: dict[str, torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray | None = None

    # Optional RL & Metadata Fields
    next_reward: torch.Tensor | np.ndarray | None = None
    next_success: bool | None = None
    episode_index: torch.Tensor | np.ndarray | None = None
    frame_index: torch.Tensor | np.ndarray | None = None
    index: torch.Tensor | np.ndarray | None = None
    task_index: torch.Tensor | np.ndarray | None = None
    timestamp: torch.Tensor | np.ndarray | None = None
    info: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


class NormalizationType(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


@dataclass(frozen=True)
class NormalizationParameters:
    """Parameters for normalizing a tensor."""
    method: NormalizationType
    mean: torch.Tensor | np.ndarray | None = None
    std: torch.Tensor | np.ndarray | None = None
    min: torch.Tensor | np.ndarray | None = None
    max: torch.Tensor | np.ndarray | None = None


@dataclass(frozen=True)
class NormalizationMap:
    """A mapping of normalization parameters for different observation components."""
    state: NormalizationParameters | None = None
    action: NormalizationParameters | None = None
    images: NormalizationParameters | None = None
    extra_normalizers: dict[str, NormalizationParameters] | None = None
    extra: dict[str, Any] | None = None

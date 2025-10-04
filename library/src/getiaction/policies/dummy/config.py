# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy config."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DummyModelConfig:
    """Configuration for DummyModel."""

    action_shape: torch.Size
    n_action_steps: int = 1
    temporal_ensemble_coeff: float | None = None
    n_obs_steps: int = 1
    horizon: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer."""

    optimizer_type: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class DummyConfig:
    """Dummy policy config with nested structure."""

    model: DummyModelConfig
    optimizer: OptimizerConfig | None = None

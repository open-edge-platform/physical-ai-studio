# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy Config."""

from dataclasses import dataclass

from physicalai.config import Config


@dataclass(frozen=True)
class PatchPolicyConfig(Config):
    """Configuration for the Patch Policy."""

    # Features
    input_features: list[str] | None = None
    output_features: list[str] | None = None

    # Steps and chunk size
    n_action_steps: int = 50
    chunk_size: int = 50
    n_obs_steps: int = 1

    # Action dimension
    action_dim: int = 32

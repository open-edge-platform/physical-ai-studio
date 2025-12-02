# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy config."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DummyConfig:
    """Configuration for a dummy policy.

    Attributes:
        action_shape (list | tuple): Shape of the action space.
        action_dtype (str | torch.dtype | None): Data type of actions.
        action_min (float | None): Minimum action value, if applicable.
        action_max (float | None): Maximum action value, if applicable.
    """

    action_shape: list | tuple
    action_dtype: str | torch.dtype | None = None
    action_min: float | None = None
    action_max: float | None = None

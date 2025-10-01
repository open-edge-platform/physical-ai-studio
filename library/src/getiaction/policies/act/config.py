# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ACT policy config."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ACTConfig:
    """ACT policy config."""

    action_shape: torch.Size | Iterable

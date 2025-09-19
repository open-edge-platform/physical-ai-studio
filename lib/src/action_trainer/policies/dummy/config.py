# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy config"""

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class DummyConfig:
    action_shape: torch.Size | Iterable

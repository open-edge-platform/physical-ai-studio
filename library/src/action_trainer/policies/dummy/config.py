# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dummy policy config."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DummyConfig:
    """Dummy policy config."""

    action_shape: torch.Size | Iterable

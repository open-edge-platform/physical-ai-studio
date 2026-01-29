# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device utilities - requires getiaction[torch]."""

from __future__ import annotations

from .utils import (
    get_available_device,
    get_device,
    get_device_count,
    get_device_name,
    is_accelerator_available,
    move_to_device,
)
from .xpu.accelerator import XPUAccelerator
from .xpu.strategy import SingleXPUStrategy

__all__ = [
    "SingleXPUStrategy",
    "XPUAccelerator",
    "get_available_device",
    "get_device",
    "get_device_count",
    "get_device_name",
    "is_accelerator_available",
    "move_to_device",
]

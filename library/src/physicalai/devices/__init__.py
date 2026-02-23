# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Devices managing utilities for physicalai."""

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

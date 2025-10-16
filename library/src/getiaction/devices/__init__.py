# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Devices managing utilities for getiaction."""

from .utils import is_xpu_available
from .xpu.accelerator import XPUAccelerator
from .xpu.strategy import SingleXPUStrategy

__all__ = ["SingleXPUStrategy", "XPUAccelerator", "is_xpu_available"]

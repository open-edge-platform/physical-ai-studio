# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XPU device Lightning modules."""

from .accelerator import XPUAccelerator
from .strategy import SingleXPUStrategy

__all__ = ["SingleXPUStrategy", "XPUAccelerator"]

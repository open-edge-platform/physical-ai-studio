# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getiaction package."""

from .devices import SingleXPUStrategy, XPUAccelerator  # register the xpu utils
from .train import Trainer

__all__ = ["SingleXPUStrategy", "Trainer", "XPUAccelerator"]
__version__ = "0.1.0"

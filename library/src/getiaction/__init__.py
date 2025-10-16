# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getiaction package."""

from .devices import is_xpu_available  # register XPU accelerator
from .train import Trainer

__all__ = ["Trainer", "is_xpu_available"]
__version__ = "0.1.0"

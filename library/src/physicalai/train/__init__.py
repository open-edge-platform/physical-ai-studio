# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI trainer."""

from importlib.metadata import version

import physicalai.devices.xpu  # noqa: F401 - ensure xpu device is registered if available

from .trainer import Trainer

__version__ = version("physicalai-train")

__all__ = ["Trainer", "__version__"]

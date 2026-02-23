# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI trainer."""

from importlib.metadata import version

from .trainer import Trainer

__version__ = version("physicalai-train")

__all__ = ["Trainer", "__version__"]

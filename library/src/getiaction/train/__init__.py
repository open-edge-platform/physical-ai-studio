# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action trainer."""

from .callbacks import VideoLogger
from .trainer import Trainer

__all__ = ["Trainer", "VideoLogger"]

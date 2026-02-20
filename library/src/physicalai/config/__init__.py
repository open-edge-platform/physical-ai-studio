# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration utilities for physicalai."""

from physicalai.config.base import Config
from physicalai.config.instantiate import instantiate_obj
from physicalai.config.mixin import FromConfig

__all__ = ["Config", "FromConfig", "instantiate_obj"]

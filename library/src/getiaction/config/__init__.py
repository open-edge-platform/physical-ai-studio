# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration utilities for getiaction."""

from getiaction.config.instantiate import instantiate_obj
from getiaction.config.mixin import FromConfig

__all__ = ["FromConfig", "instantiate_obj"]

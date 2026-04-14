# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base classes for policies."""

from .model import Model
from .policy import Policy, PolicyLike

__all__ = ["Model", "Policy", "PolicyLike"]

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patch Policy action heads."""

from .behavior_transformer import BehaviourTransformerHead
from .diffusion import DiffusionHead

__all__ = ["BehaviourTransformerHead", "DiffusionHead"]

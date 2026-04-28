# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trossen WidowX AI robot manipulator driver package.

Public API::

    from physicalai.robot.trossen import WidowXAI, WidowXAIObservation
    from physicalai.robot.trossen import BimanualWidowXAI, BimanualWidowXAIObservation
"""

from .bimanual_widowxai import BimanualWidowXAI, BimanualWidowXAIObservation
from .widowxai import WidowXAI, WidowXAIObservation

__all__ = ["BimanualWidowXAI", "BimanualWidowXAIObservation", "WidowXAI", "WidowXAIObservation"]

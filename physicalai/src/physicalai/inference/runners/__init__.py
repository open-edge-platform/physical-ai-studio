# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference runners.

Runners encapsulate the execution pattern — how adapter calls map to
actions — and are composed into ``InferenceModel`` at construction time.
"""

from __future__ import annotations

from physicalai.inference.runners.action_chunking import ActionChunking
from physicalai.inference.runners.base import InferenceRunner
from physicalai.inference.runners.factory import get_runner
from physicalai.inference.runners.single_pass import SinglePass

__all__ = [
    "ActionChunking",
    "InferenceRunner",
    "SinglePass",
    "get_runner",
]

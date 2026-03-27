# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference postprocessors.

Postprocessors transform inference outputs after the runner returns.
"""

from physicalai.inference.postprocessors.action_normalizer import ActionNormalizer
from physicalai.inference.postprocessors.base import Postprocessor

__all__ = [
    "ActionNormalizer",
    "Postprocessor",
]

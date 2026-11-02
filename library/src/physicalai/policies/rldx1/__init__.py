# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""RLDX-1 Policy - first-party implementation.

RLDX-1 (RLWRLD) is a flow-matching Vision-Language-Action model built on a
Qwen3-VL-8B backbone and a Multi-Stream Action Transformer (MSAT) action head.

v1 scope: pre-train (PT) -> fine-tune (FT) path only, starting from
``RLWRLD/RLDX-1-PT``. The motion / memory / physics add-on streams and the
RECAP RL trainer are deferred to phase 2. See
``library/docs/rldx-1-integration.md``.

.. note::
    Upstream weights ship under the non-commercial RLWRLD Model License v1.0.
    The integration is research-only unless RLWRLD relicenses.
"""

from physicalai.policies.rldx1.config import Rldx1Config
from physicalai.policies.rldx1.model import Rldx1Model
from physicalai.policies.rldx1.policy import Rldx1
from physicalai.policies.rldx1.preprocessor import (
    Rldx1Postprocessor,
    Rldx1Preprocessor,
    make_rldx1_transforms,
)

__all__ = [
    "Rldx1",
    "Rldx1Config",
    "Rldx1Model",
    "Rldx1Postprocessor",
    "Rldx1Preprocessor",
    "make_rldx1_transforms",
]

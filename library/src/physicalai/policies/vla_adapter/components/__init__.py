# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Building blocks for the VLA-Adapter policy.

Split by role: :mod:`.vlm` is the perception stack, :mod:`.action_head` the
policy head, and :mod:`.proprio` the state projector bridging them — the same
separation upstream keeps between ``action_heads.py`` and ``projectors.py``.

The split is by role, not by what trains: the VLM's visual projector and action
queries train alongside the head, while its towers and language model stay
frozen.
"""

from physicalai.policies.vla_adapter.components.action_head import L1RegressionActionHead, MLPResNet, MLPResNetBlock
from physicalai.policies.vla_adapter.components.proprio import ProprioProjector
from physicalai.policies.vla_adapter.components.vlm import VLM, PrismaticVisionBackbone, PrismaticVisualProjector

__all__ = [
    "VLM",
    "L1RegressionActionHead",
    "MLPResNet",
    "MLPResNetBlock",
    "PrismaticVisionBackbone",
    "PrismaticVisualProjector",
    "ProprioProjector",
]

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA Policy - Vision-Language-Action model with a latent video world model.

The architecture modules live under :mod:`physicalai.policies.vla_jepa.components` (Qwen3-VL
backbone, DiT flow-matching action head, V-JEPA2 action-conditioned video predictor), mirroring the
layout of the other first-party families.
"""

from physicalai.policies.vla_jepa.components import (
    ActionConditionedVideoPredictor,
    Qwen3VLInterface,
    VLAJEPAActionHead,
)
from physicalai.policies.vla_jepa.config import VLAJEPAConfig
from physicalai.policies.vla_jepa.model import VLAJEPAModel
from physicalai.policies.vla_jepa.policy import VLAJEPA
from physicalai.policies.vla_jepa.preprocessor import (
    VLAJEPAPostprocessor,
    VLAJEPAPreprocessor,
    make_vla_jepa_preprocessors,
)

__all__ = [
    "VLAJEPA",
    "ActionConditionedVideoPredictor",
    "Qwen3VLInterface",
    "VLAJEPAActionHead",
    "VLAJEPAConfig",
    "VLAJEPAModel",
    "VLAJEPAPostprocessor",
    "VLAJEPAPreprocessor",
    "make_vla_jepa_preprocessors",
]

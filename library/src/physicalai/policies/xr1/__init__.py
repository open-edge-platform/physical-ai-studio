# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR1 (Xiaomi-Robotics-1) vision-language-action policy.

XR1 couples a Qwen3-VL backbone with a DiT action expert in a
Mixture-of-Transformers layout and is trained with flow matching.

Reference: `Xiaomi-Robotics-1 <https://arxiv.org/abs/2607.15330>`_.
"""

from .config import XR1Config
from .embodiment import ACTION_SLOTS, ALOHA_STATE_TO_XR1, STATE_SLOTS
from .graph_export import (
    ActionExpertInputs,
    XR1ActionExpert,
    build_action_expert_inputs,
    export_action_expert,
)
from .policy import XR1
from .preprocessor import XR1Postprocessor, XR1Preprocessor, make_xr1_preprocessors
from .pretrained_utils import (
    EmbodimentStats,
    infer_config_overrides,
    load_pretrained_weights,
    load_state_dict,
    read_embodiment_stats,
)
from .vla import XR1Model

__all__ = [
    "ACTION_SLOTS",
    "ALOHA_STATE_TO_XR1",
    "STATE_SLOTS",
    "XR1",
    "ActionExpertInputs",
    "EmbodimentStats",
    "XR1ActionExpert",
    "XR1Config",
    "XR1Model",
    "XR1Postprocessor",
    "XR1Preprocessor",
    "build_action_expert_inputs",
    "export_action_expert",
    "infer_config_overrides",
    "load_pretrained_weights",
    "load_state_dict",
    "make_xr1_preprocessors",
    "read_embodiment_stats",
]

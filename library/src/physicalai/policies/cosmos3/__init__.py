# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Cosmos 3 Policy.

Multimodal world model policy based on diffusers Cosmos3OmniPipeline and rectified flow matching.
"""

from .config import Cosmos3Config
from .flow_matching import build_action_tokens, build_pack, flow_matching_step
from .model import Cosmos3Model
from .pipeline import (
    DEFAULT_MIN_XPU_DRIVER,
    PolicyPipelineWithState,
    check_xpu_driver,
    require_xpu_driver,
    state_action_mrope_ids,
)
from .policy import Cosmos3
from .surgery import (
    GEN_TOWER_KEYS,
    HEAD_KEYS,
    LORA_TARGETS,
    configure_trainable,
    init_domain_action_head,
    load_finetuned,
    split_trainable_params,
)

__all__ = [
    "DEFAULT_MIN_XPU_DRIVER",
    "GEN_TOWER_KEYS",
    "HEAD_KEYS",
    "LORA_TARGETS",
    "Cosmos3",
    "Cosmos3Config",
    "Cosmos3Model",
    "PolicyPipelineWithState",
    "build_action_tokens",
    "build_pack",
    "check_xpu_driver",
    "configure_trainable",
    "flow_matching_step",
    "init_domain_action_head",
    "load_finetuned",
    "require_xpu_driver",
    "split_trainable_params",
    "state_action_mrope_ids",
]

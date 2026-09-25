# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Cosmos 3 Policy.

Multimodal world model policy based on diffusers Cosmos3OmniPipeline and rectified flow matching.
"""

from .config import DEFAULT_COSMOS3_REVISION, Cosmos3Config
from .flow_matching import build_action_tokens, build_pack, flow_matching_step
from .model import Cosmos3Model
from .normalization import load_stats_file, resolve_affine
from .pipeline import (
    DEFAULT_MIN_XPU_DRIVER,
    PolicyPipelineWithState,
    check_xpu_driver,
    require_xpu_driver,
    state_action_mrope_ids,
)
from .policy import Cosmos3
from .preprocessor import Cosmos3Preprocessor, compose_horizontal_views, compose_t_views
from .representation import (
    EMBODIMENT_ACTION_SPACE,
    EMBODIMENT_NORMALIZATION,
    embodiment_gripper_flipped,
    embodiment_normalization,
    flip_gripper_last_channel,
    resolve_action_space,
    uses_minmax_normalization,
)
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
    "DEFAULT_COSMOS3_REVISION",
    "DEFAULT_MIN_XPU_DRIVER",
    "EMBODIMENT_ACTION_SPACE",
    "EMBODIMENT_NORMALIZATION",
    "GEN_TOWER_KEYS",
    "HEAD_KEYS",
    "LORA_TARGETS",
    "Cosmos3",
    "Cosmos3Config",
    "Cosmos3Model",
    "Cosmos3Preprocessor",
    "PolicyPipelineWithState",
    "build_action_tokens",
    "build_pack",
    "check_xpu_driver",
    "compose_horizontal_views",
    "compose_t_views",
    "configure_trainable",
    "embodiment_gripper_flipped",
    "embodiment_normalization",
    "flip_gripper_last_channel",
    "flow_matching_step",
    "init_domain_action_head",
    "load_finetuned",
    "load_stats_file",
    "require_xpu_driver",
    "resolve_action_space",
    "resolve_affine",
    "split_trainable_params",
    "state_action_mrope_ids",
    "uses_minmax_normalization",
]

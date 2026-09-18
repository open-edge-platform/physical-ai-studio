# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Cosmos 3 Policy.

Multimodal world model policy based on diffusers Cosmos3OmniPipeline and rectified flow matching.
"""

from .config import Cosmos3Config
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
    DOMAIN_NORMALIZATION,
    DOMAIN_REPRESENTATION,
    assemble_state_sequence,
    domain_normalization,
    domain_representation,
    represent_actions,
    represent_state,
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
    "DEFAULT_MIN_XPU_DRIVER",
    "DOMAIN_NORMALIZATION",
    "DOMAIN_REPRESENTATION",
    "GEN_TOWER_KEYS",
    "HEAD_KEYS",
    "LORA_TARGETS",
    "Cosmos3",
    "Cosmos3Config",
    "Cosmos3Model",
    "Cosmos3Preprocessor",
    "PolicyPipelineWithState",
    "assemble_state_sequence",
    "build_action_tokens",
    "build_pack",
    "check_xpu_driver",
    "compose_horizontal_views",
    "compose_t_views",
    "configure_trainable",
    "domain_normalization",
    "domain_representation",
    "flow_matching_step",
    "init_domain_action_head",
    "load_finetuned",
    "load_stats_file",
    "represent_actions",
    "represent_state",
    "require_xpu_driver",
    "resolve_affine",
    "split_trainable_params",
    "state_action_mrope_ids",
    "uses_minmax_normalization",
]

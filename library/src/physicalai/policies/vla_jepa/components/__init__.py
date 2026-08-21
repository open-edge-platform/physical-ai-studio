# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VLA-JEPA model components - pure PyTorch implementations.

This module contains the core neural network components for the VLA-JEPA model:

- `qwen_interface`: Qwen3-VL vision-language backbone, action-token vocabulary and prompting
- `action_head`: DiT flow-matching action head predicting the action chunk
- `world_model`: V-JEPA2 action-conditioned video predictor used as a training-only auxiliary loss

The file layout mirrors LeRobot's ``lerobot.policies.vla_jepa`` package so upstream changes stay
diffable, and the class and attribute names are preserved verbatim because they determine the keys
of the published checkpoints.

Note:
    `action_head` requires `diffusers` (``DiT`` subclasses ``ModelMixin``/``ConfigMixin``) and
    `qwen_interface` requires `transformers` to load the Qwen3-VL backbone from HuggingFace. Both
    ship in the ``vla_jepa`` extra.
"""

from physicalai.policies.vla_jepa.components.action_head import (
    DIT_PRESETS,
    ActionEncoder,
    ActionModelPreset,
    AdaLayerNorm,
    BasicTransformerBlock,
    DiT,
    SinusoidalPositionalEncoding,
    TimestepEncoder,
    VLAJEPAActionHead,
)
from physicalai.policies.vla_jepa.components.qwen_interface import Qwen3VLInterface, resolve_torch_dtype
from physicalai.policies.vla_jepa.components.world_model import (
    MLP,
    ACBlock,
    ACRoPEAttention,
    ActionConditionedVideoPredictor,
    DropPath,
    build_action_block_causal_attention_mask,
    rotate_queries_or_keys,
)

__all__ = [
    "DIT_PRESETS",
    "MLP",
    "ACBlock",
    "ACRoPEAttention",
    "ActionConditionedVideoPredictor",
    "ActionEncoder",
    "ActionModelPreset",
    "AdaLayerNorm",
    "BasicTransformerBlock",
    "DiT",
    "DropPath",
    "Qwen3VLInterface",
    "SinusoidalPositionalEncoding",
    "TimestepEncoder",
    "VLAJEPAActionHead",
    "build_action_block_causal_attention_mask",
    "resolve_torch_dtype",
    "rotate_queries_or_keys",
]

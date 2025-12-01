# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot model components - pure PyTorch implementations.

This module contains the core neural network components for the GR00T-N1.5 model:

- `backbone`: Eagle VLM backbone and processor for vision-language encoding
- `transformer`: Diffusion Transformer (DiT) with cross-attention
- `action_head`: Flow matching action head for action prediction
- `nn`: Reusable neural network primitives (encodings, layers)

These components are designed to be:
- Framework-agnostic (pure PyTorch, no LeRobot dependency)
- Device-flexible (CUDA, XPU via SDPA attention)
- Reusable for other VLA policies

Note:
    The FlowMatchingActionHead uses `diffusers` for some attention utilities.
    The Eagle backbone requires `transformers` for loading the
    pretrained Eagle2 model from HuggingFace.
"""

from getiaction.policies.groot.components.action_head import (
    FlowMatchingActionHead,
    FlowMatchingActionHeadConfig,
)
from getiaction.policies.groot.components.backbone import EagleBackbone, EagleProcessor
from getiaction.policies.groot.components.nn import (
    CategorySpecificLinear,
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
    swish,
)
from getiaction.policies.groot.components.transformer import (
    AdaLayerNorm,
    BasicTransformerBlock,
    TimestepEncoder,
    get_dit_class,
    get_self_attention_transformer_class,
)

__all__ = [
    "AdaLayerNorm",
    "BasicTransformerBlock",
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "EagleBackbone",
    "EagleProcessor",
    "FlowMatchingActionHead",
    "FlowMatchingActionHeadConfig",
    "MultiEmbodimentActionEncoder",
    "SinusoidalPositionalEncoding",
    "TimestepEncoder",
    "get_dit_class",
    "get_self_attention_transformer_class",
    "swish",
]

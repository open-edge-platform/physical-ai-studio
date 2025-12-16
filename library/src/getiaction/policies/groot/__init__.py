# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Groot (GR00T-N1.5) Policy - First-party implementation.

This module provides a first-party Lightning policy for NVIDIA's GR00T-N1.5-3B
foundation model, with several improvements:

1. **Explicit hyperparameters** - All args visible in __init__ for discoverability
2. **Native Lightning checkpoints** - load_from_checkpoint() just works
3. **PyTorch SDPA attention** - No Flash Attention CUDA dependency, works on XPU
4. **Separate nn.Module** - Model can be used standalone for export/testing
5. **No LeRobot dependency** - Pure PyTorch/HuggingFace implementation

## Quick Start

```python
from getiaction.policies.groot import Groot, GrootModel
from getiaction.train import Trainer

# Training with Lightning
policy = Groot(
    chunk_size=50,
    attn_implementation='sdpa',  # PyTorch native, no flash-attn needed
    tune_projector=True,
)
trainer = Trainer(max_epochs=100)
trainer.fit(policy, datamodule)

# Load checkpoint (native Lightning)
policy = Groot.load_from_checkpoint("checkpoint.ckpt")

# Standalone model usage
model = GrootModel.from_pretrained(
    "nvidia/GR00T-N1.5-3B",
    attn_implementation='sdpa',
)
actions = model.get_action(batch)

# From config file
model = GrootModel.from_config("config.yaml")
```

## Attention Implementations

- `sdpa` (default): PyTorch native SDPA - works on CUDA and XPU
- `flash_attention_2`: Requires flash-attn CUDA package
- `eager`: Fallback Python implementation

## Dependencies

Required:
- transformers: For Eagle VLM model loading
- huggingface_hub: For downloading pretrained weights
- diffusers: For attention and embedding components in action head
"""

from getiaction.policies.groot.components import EagleBackbone, EagleProcessor
from getiaction.policies.groot.config import GrootConfig
from getiaction.policies.groot.model import GrootModel
from getiaction.policies.groot.policy import Groot
from getiaction.policies.groot.preprocessor import (
    GrootPostprocessor,
    GrootPreprocessor,
    make_groot_preprocessors,
)

__all__ = [
    "EagleBackbone",
    "EagleProcessor",
    "Groot",
    "GrootConfig",
    "GrootModel",
    "GrootPostprocessor",
    "GrootPreprocessor",
    "make_groot_preprocessors",
]

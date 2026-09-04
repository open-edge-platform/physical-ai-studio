# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR0 Policy - Xiaomi's flow matching VLA model (Qwen3-VL-4B + DiT action head)."""

from __future__ import annotations

from .config import XR0Config
from .dit import XR0FlowModel
from .model import XR0Model
from .policy import XR0
from .preprocessor import (
    XR0Postprocessor,
    XR0Preprocessor,
    make_xr0_preprocessors,
)
from .pretrained_utils import (
    extract_xr0_dataset_stats,
    load_xr0_pretrained_weights,
    remap_xr0_state_dict,
    resolve_pretrained_path,
)
from .qwen3_vlm import XR0Qwen3VL

__all__ = [
    "XR0",
    "XR0Config",
    "XR0FlowModel",
    "XR0Model",
    "XR0Postprocessor",
    "XR0Preprocessor",
    "XR0Qwen3VL",
    "extract_xr0_dataset_stats",
    "load_xr0_pretrained_weights",
    "make_xr0_preprocessors",
    "remap_xr0_state_dict",
    "resolve_pretrained_path",
]

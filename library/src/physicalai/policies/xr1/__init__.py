# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""XR1 (Xiaomi-Robotics-1) vision-language-action policy.

XR1 couples a Qwen3-VL backbone with a DiT action expert in a
Mixture-of-Transformers layout and is trained with flow matching.

Reference: `Xiaomi-Robotics-1 <https://arxiv.org/abs/2607.15330>`_.
"""

from .config import XR1Config
from .policy import XR1
from .preprocessor import XR1Postprocessor, XR1Preprocessor, make_xr1_preprocessors
from .vla import XR1Model

__all__ = [
    "XR1",
    "XR1Config",
    "XR1Model",
    "XR1Postprocessor",
    "XR1Preprocessor",
    "make_xr1_preprocessors",
]

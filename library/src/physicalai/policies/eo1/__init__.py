# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EO-1 Policy - interleaved vision-text-action model with a flow-matching action head.

EO-1 formats each robot-control sample as a multimodal conversation for a Qwen2.5-VL backbone:
camera frames go in as images, the robot state occupies a state placeholder token and the future
action chunk occupies `chunk_size` action placeholder tokens that a continuous flow-matching head
denoises.

Example:
    >>> from physicalai.policies.eo1 import EO1, EO1Config
    >>> policy = EO1(chunk_size=8, n_action_steps=8)
"""

from .config import EO1Config
from .model import EO1Model
from .policy import EO1

__all__ = ["EO1", "EO1Config", "EO1Model"]

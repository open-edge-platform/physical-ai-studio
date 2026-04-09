# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation field name constants for convenient dict access.

Module-level constants providing string literals for Observation field names,
enabling IDE autocomplete and safe refactoring for dict-based access patterns.
"""

from physicalai.inference.constants import ACTION, IMAGE_MASKS, IMAGES, TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK

# Core observation fields
STATE = "state"

# Optional RL & metadata fields
NEXT_REWARD = "next_reward"
NEXT_SUCCESS = "next_success"
EPISODE_INDEX = "episode_index"
FRAME_INDEX = "frame_index"
INDEX = "index"
TASK_INDEX = "task_index"
TIMESTAMP = "timestamp"
INFO = "info"
EXTRA = "extra"

__all__ = [
    "ACTION",
    "EPISODE_INDEX",
    "EXTRA",
    "FRAME_INDEX",
    "IMAGES",
    "IMAGE_MASKS",
    "INDEX",
    "INFO",
    "NEXT_REWARD",
    "NEXT_SUCCESS",
    "STATE",
    "TASK",
    "TASK_INDEX",
    "TIMESTAMP",
    "TOKENIZED_PROMPT",
    "TOKENIZED_PROMPT_MASK",
]

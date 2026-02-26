# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation field name constants for convenient dict access.

Module-level constants providing string literals for Observation field names,
enabling IDE autocomplete and safe refactoring for dict-based access patterns.

Usage:
    from physicalai.data.constants import STATE, ACTION, IMAGES
    from physicalai.data.observation import Observation

    # Dict-based access with constants
    batch[STATE]   # equivalent to batch["state"]
    batch[ACTION]  # equivalent to batch["action"]

    # All of these are equivalent:
    batch[ACTION]                              # imported constant (recommended)
    batch["action"]                            # string literal
    batch[Observation.FieldName.ACTION]        # enum member

Note:
    These constants are torch-free and can be imported without loading torch.
    This module is designed for use in data pipelines that need to avoid
    torch imports until necessary.
"""

# Core observation fields
ACTION = "action"
TASK = "task"
STATE = "state"
IMAGES = "images"

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
    "INDEX",
    "INFO",
    "NEXT_REWARD",
    "NEXT_SUCCESS",
    "STATE",
    "TASK",
    "TASK_INDEX",
    "TIMESTAMP",
]

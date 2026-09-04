# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared key constants for RLDX-1 tensor-dict contracts.

These keys are exchanged between the RLDX-1 preprocessor, backbone, action
model, graph-safe export wrapper, and policy export pipeline. Centralizing them
avoids drift across modules.
"""

INPUT_IDS = "input_ids"
IMAGE_GRID_THW = "image_grid_thw"
BACKBONE_FEATURES = "backbone_features"
ACTION_PRED = "action_pred"
PIXEL_VALUES = "pixel_values"
STATE = "state"
EMBODIMENT_ID = "embodiment_id"
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"

__all__ = [
    "ACTION_PRED",
    "ATTENTION_MASK",
    "BACKBONE_FEATURES",
    "EMBODIMENT_ID",
    "IMAGE_GRID_THW",
    "INPUT_IDS",
    "PIXEL_VALUES",
    "POSITION_IDS",
    "STATE",
]

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production inference module for exported policies.

This module provides a unified interface for running inference with
exported policies across different backends (OpenVINO, ONNX, Torch Export IR).

Key Features:
    - Unified API matching PyTorch policies
    - Auto-detection of backend and device
    - Support for chunked/stateful policies
    - Handles action queues automatically

Examples:
    >>> from getiaction.inference import InferenceModel
    >>> policy = InferenceModel.load("./exports/act_policy")
    >>> policy.reset()
    >>> action = policy.select_action({"state": state_array, "images": images_array})
"""

from .model import ExportBackend, InferenceModel

__all__ = ["ExportBackend", "InferenceModel"]

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Production inference module for exported policies.

This module provides a unified interface for running inference with
exported policies across different backends (OpenVINO, ONNX, TorchScript).

Key Features:
    - Unified API matching PyTorch policies
    - Auto-detection of backend and device
    - Support for chunked/stateful policies
    - Handles action queues automatically

Examples:
    >>> from getiaction.inference import InferenceModel
    >>> # Load with auto-detection
    >>> policy = InferenceModel.load("./exports/act_policy")

    >>> # Use like PyTorch policy
    >>> policy.reset()
    >>> action = policy.select_action(observation)
"""

from getiaction.inference.model import InferenceModel

__all__ = ["InferenceModel"]

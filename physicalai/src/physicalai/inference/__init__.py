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
    - Lifecycle callbacks for instrumentation

Examples:
    >>> from physicalai.inference import InferenceModel
    >>> # Load with auto-detection
    >>> policy = InferenceModel.load("./exports/act_policy")

    >>> # Use like PyTorch policy
    >>> policy.reset()
    >>> action = policy.select_action(observation)
"""

# Allow ``physicalai.inference`` to be extended by other distributions sharing
# the ``physicalai`` namespace (e.g. the ``physicalai-train`` library, which
# ships the torch and executorch adapters under ``physicalai.inference.adapters``).
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from physicalai.inference.callbacks.base import Callback
from physicalai.inference.model import InferenceModel

__all__ = ["Callback", "InferenceModel"]

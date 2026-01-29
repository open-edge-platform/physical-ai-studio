# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export types - torch-free enums and constants.

This module contains types used for export configuration that can be
imported without requiring torch dependencies.
"""

from __future__ import annotations

from enum import StrEnum


class ExportBackend(StrEnum):
    """Supported export backends.

    Used to specify which backend to use for model export and inference.

    Attributes:
        ONNX: ONNX Runtime backend (~50MB)
        OPENVINO: Intel OpenVINO backend (~100MB)
        TORCH: PyTorch backend (for checkpoints)
        TORCH_EXPORT_IR: PyTorch Export IR backend
    """

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"
    TORCH_EXPORT_IR = "torch_export_ir"


__all__ = ["ExportBackend"]

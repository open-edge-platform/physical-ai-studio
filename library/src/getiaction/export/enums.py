# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enums for export module."""

from enum import StrEnum


class ExportBackend(StrEnum):
    """Supported export backends for model inference."""

    OPENVINO = "openvino"
    ONNX = "onnx"
    TORCH = "torch"
    TORCH_EXPORT_IR = "torch_export_ir"

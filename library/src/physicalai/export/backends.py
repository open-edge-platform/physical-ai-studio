# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export backends enumeration."""

from enum import StrEnum


class ExportBackend(StrEnum):
    """Supported export backends."""

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"
    TORCH_EXPORT_IR = "torch_export_ir"


__all__ = ["ExportBackend"]

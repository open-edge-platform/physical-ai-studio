# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export backends enumeration and parameters."""

from dataclasses import dataclass, field
from enum import StrEnum


class ExportBackend(StrEnum):
    """Supported export backends."""

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"
    TORCH_EXPORT_IR = "torch_export_ir"

    @property
    def extension(self) -> str:
        """Canonical file extension for this backend (including leading dot)."""
        extensions = {
            "onnx": ".onnx",
            "openvino": ".xml",
            "torch": ".pt",
            "torch_export_ir": ".pt2",
        }
        return extensions[self.value]

    @property
    def parameter_class(self) -> type["ExportParameters"]:
        """The class of export parameters for this backend."""
        parameter_classes = {
            "onnx": ONNXExportParameters,
            "openvino": OpenVINOExportParameters,
            "torch": TorchExportParameters,
            "torch_export_ir": ExportParameters,
        }
        return parameter_classes[self.value]


@dataclass
class ExportParameters:
    """Parameters for exporting a model."""

    exporter_kwargs: dict = field(default_factory=dict)
    preprocessing_type: str = ""


@dataclass
class ONNXExportParameters(ExportParameters):
    """Parameters specific to ONNX export."""

    export_tokenizer: bool = False


@dataclass
class OpenVINOExportParameters(ExportParameters):
    """Parameters specific to OpenVINO export."""

    export_tokenizer: bool = False
    outputs: list[str] = field(default_factory=lambda: ["action"])
    compress_to_fp16: bool = False
    via_onnx: bool = False


@dataclass
class TorchExportParameters(ExportParameters):
    """Parameters specific to torch export."""

    input_names: list[str] = field(default_factory=lambda: ["observation"])
    output_names: list[str] = field(default_factory=lambda: ["action"])


__all__ = [
    "ExportBackend",
    "ExportParameters",
    "ONNXExportParameters",
    "OpenVINOExportParameters",
    "TorchExportParameters",
]

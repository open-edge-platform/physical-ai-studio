# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export backends enumeration and parameters."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

#: Supported ExecuTorch delegate backends.
ExecuTorchDelegate = Literal["portable", "xnnpack", "openvino"]


class ExportBackend(StrEnum):
    """Supported export backends."""

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"
    EXECUTORCH = "executorch"

    @property
    def extension(self) -> str:
        """Canonical file extension for this backend (including leading dot)."""
        extensions = {
            "onnx": ".onnx",
            "openvino": ".xml",
            "torch": ".pt",
            "executorch": ".pte",
        }
        return extensions[self.value]

    @property
    def parameter_class(self) -> type["ExportParameters"]:
        """The class of export parameters for this backend."""
        parameter_classes = {
            "onnx": ONNXExportParameters,
            "openvino": OpenVINOExportParameters,
            "torch": TorchExportParameters,
            "executorch": ExecuTorchExportParameters,
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


@dataclass
class ExecuTorchExportParameters(ExportParameters):
    """Parameters specific to ExecuTorch export.

    Attributes:
        delegate: The delegate backend to use for ExecuTorch export.
            Supported values: ``"portable"`` (default), ``"xnnpack"``, ``"openvino"``.
        output_names: Names for model outputs stored in metadata for inference.
    """

    delegate: ExecuTorchDelegate = "portable"
    output_names: list[str] = field(default_factory=lambda: ["action"])


__all__ = [
    "ExecuTorchDelegate",
    "ExecuTorchExportParameters",
    "ExportBackend",
    "ExportParameters",
    "ONNXExportParameters",
    "OpenVINOExportParameters",
    "TorchExportParameters",
]

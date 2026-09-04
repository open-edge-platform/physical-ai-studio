# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export backends enumeration and parameters."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
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
    """Parameters for exporting a model.

    Attributes:
        exporter_kwargs: Extra keyword arguments forwarded to the backend exporter.
        preprocessors_specs: Component specs describing the inference preprocessors
            to record in the manifest.
        postprocessors_specs: Component specs describing the inference postprocessors
            to record in the manifest.
        pre_export_hooks: Callables invoked in order right before the model is
            traced/converted. Use them to mutate the model in place for export
            (e.g. bake constants into the graph, swap ops for export-friendly
            variants). Signature: ``() -> object`` (any return value is ignored).
        post_export_hooks: Callables invoked in order after the model has been
            written to disk. Each hook receives the path to the exported file
            and may modify the file in place (e.g. to patch the graph for a
            specific runtime). Signature: ``(export_path: str | Path) -> object``
            (any return value is ignored).
    """

    exporter_kwargs: dict = field(default_factory=dict)
    preprocessors_specs: list = field(default_factory=list)
    postprocessors_specs: list = field(default_factory=list)
    pre_export_hooks: list[Callable[[], object]] = field(default_factory=list)
    post_export_hooks: list[Callable[[str | Path], object]] = field(default_factory=list)


@dataclass
class ONNXExportParameters(ExportParameters):
    """Parameters specific to ONNX export.

    Attributes:
        export_tokenizer: When ``True``, request tokenizer export. Not supported for
            the ONNX backend; setting it raises at export time.
    """

    export_tokenizer: bool = False


@dataclass
class OpenVINOExportParameters(ExportParameters):
    """Parameters specific to OpenVINO export.

    Attributes:
        export_tokenizer: When ``True``, convert the preprocessor's tokenizer to an
            OpenVINO tokenizer and save it alongside the model as ``tokenizer.xml``.
        outputs: Ordered names to assign to the converted model's output tensors.
        compress_to_fp16: When ``True``, compress the saved model's floating-point
            constants to FP16. When ``False``, weights are kept at their original
            precision.
        via_onnx: When ``True``, export to a temporary ONNX file first and convert
            that to OpenVINO, instead of converting the torch model directly.
        input_name_map: Optional mapping ``{traced_input_name: exported_name}`` used
            to rename the converted graph's input tensors before saving. Useful to
            align the graph ports with the keys emitted by preprocessor components
            (e.g. an ``ov_tokenizer`` producing ``tokenized_prompt``).
    """

    export_tokenizer: bool = False
    outputs: list[str] = field(default_factory=lambda: ["action"])
    compress_to_fp16: bool = False
    via_onnx: bool = False
    input_name_map: dict[str, str] = field(default_factory=dict)


@dataclass
class TorchExportParameters(ExportParameters):
    """Parameters specific to torch export.

    Attributes:
        input_names: Names of the model inputs recorded in the manifest for inference.
        output_names: Names of the model outputs recorded in the manifest for inference.
    """

    input_names: list[str] = field(default_factory=lambda: ["observation"])
    output_names: list[str] = field(default_factory=lambda: ["action"])


@dataclass
class ExecuTorchExportParameters(ExportParameters):
    """Parameters specific to ExecuTorch export.

    Attributes:
        delegate: The delegate backend to use for ExecuTorch export.
            Supported values: ``"portable"`` (default), ``"xnnpack"``, ``"openvino"``.
        output_names: Names for model outputs stored in the manifest for inference.
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

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for different backend runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from getiaction.inference.adapters.base import RuntimeAdapter
from getiaction.inference.adapters.onnx import ONNXAdapter
from getiaction.inference.adapters.openvino import OpenVINOAdapter
from getiaction.inference.adapters.torch import TorchAdapter
from getiaction.inference.adapters.torch_export import TorchExportAdapter

if TYPE_CHECKING:
    from getiaction.export import ExportBackend

__all__ = [
    "ONNXAdapter",
    "OpenVINOAdapter",
    "RuntimeAdapter",
    "TorchAdapter",
    "TorchExportAdapter",
    "get_adapter",
]


def get_adapter(backend: ExportBackend | str, **kwargs: Any) -> RuntimeAdapter:  # noqa: ARG001, ANN401
    """Get the appropriate adapter for a given backend.

    Args:
        backend: The export backend (ExportBackend enum or string)
        **kwargs: Additional adapter configuration (currently unused but accepted for compatibility)

    Returns:
        Instantiated adapter for the backend

    Raises:
        ValueError: If backend is not supported

    Examples:
        >>> from getiaction.export import ExportBackend
        >>> adapter = get_adapter(ExportBackend.OPENVINO)
        >>> adapter.load(Path("model.xml"))
        >>> # Can also use string
        >>> adapter = get_adapter("openvino")
    """
    from getiaction.export import ExportBackend  # noqa: PLC0415

    # Convert string to enum if needed
    if isinstance(backend, str):
        backend = ExportBackend(backend)

    adapter_map: dict[ExportBackend, type[RuntimeAdapter]] = {
        ExportBackend.OPENVINO: OpenVINOAdapter,
        ExportBackend.ONNX: ONNXAdapter,
        ExportBackend.TORCH_EXPORT_IR: TorchExportAdapter,
        ExportBackend.TORCH: TorchAdapter,
    }

    if backend not in adapter_map:
        msg = f"No adapter available for backend: {backend}"
        raise ValueError(msg)

    return adapter_map[backend]()

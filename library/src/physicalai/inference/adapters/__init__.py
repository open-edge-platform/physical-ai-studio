# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for different backend runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.export.backends import ExportBackend

from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.onnx import ONNXAdapter
from physicalai.inference.adapters.openvino import OpenVINOAdapter

__all__ = [
    "ONNXAdapter",
    "OpenVINOAdapter",
    "RuntimeAdapter",
    "TorchAdapter",  # noqa: F822
    "TorchExportAdapter",  # noqa: F822
    "get_adapter",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load torch-dependent adapters on access.

    Args:
        name: Attribute name to load.

    Returns:
        The requested adapter class.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name == "TorchAdapter":
        from physicalai.inference.adapters.torch import TorchAdapter  # noqa: PLC0415

        return TorchAdapter
    if name == "TorchExportAdapter":
        from physicalai.inference.adapters.torch_export import TorchExportAdapter  # noqa: PLC0415

        return TorchExportAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
        >>> from physicalai.export.backends import ExportBackend
        >>> adapter = get_adapter(ExportBackend.OPENVINO)
        >>> adapter.load(Path("model.xml"))
        >>> # Can also use string
        >>> adapter = get_adapter("openvino")
    """
    from physicalai.export.backends import ExportBackend  # noqa: PLC0415

    # Convert string to enum if needed
    if isinstance(backend, str):
        backend = ExportBackend(backend)

    adapter_map: dict[ExportBackend, type[RuntimeAdapter]] = {
        ExportBackend.OPENVINO: OpenVINOAdapter,
        ExportBackend.ONNX: ONNXAdapter,
    }

    # Lazy-import torch adapters only when needed
    if backend in {ExportBackend.TORCH, ExportBackend.TORCH_EXPORT_IR}:
        from physicalai.inference.adapters.torch import TorchAdapter  # noqa: PLC0415
        from physicalai.inference.adapters.torch_export import TorchExportAdapter  # noqa: PLC0415

        adapter_map[ExportBackend.TORCH] = TorchAdapter
        adapter_map[ExportBackend.TORCH_EXPORT_IR] = TorchExportAdapter

    if backend not in adapter_map:
        msg = f"No adapter available for backend: {backend}"
        raise ValueError(msg)

    return adapter_map[backend]()

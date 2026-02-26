# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export mixins module."""

from typing import Any

from .backends import ExportBackend


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "Export":
        from .mixin_export import Export  # noqa: PLC0415

        return Export
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def get_available_backends() -> list[str]:
    """Get list of available export backends.

    Returns:
        List of backend names as strings.

    Examples:
        >>> from physicalai.export import get_available_backends
        >>> backends = get_available_backends()
        >>> print(backends)
        ['onnx', 'openvino', 'torch', 'torch_export_ir']
    """
    return [backend.value for backend in ExportBackend]


__all__ = ["Export", "ExportBackend", "get_available_backends"]  # noqa: F822

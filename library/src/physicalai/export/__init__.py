# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export mixins module."""

from .backends import ExportBackend


def __getattr__(name):
    if name == "Export":
        from .mixin_export import Export
        return Export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


__all__ = ["Export", "ExportBackend", "get_available_backends"]

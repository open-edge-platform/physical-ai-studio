# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export mixins module."""

from .backends import ExportBackend
from .mixin_export import Export


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

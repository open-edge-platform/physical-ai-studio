# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export mixins module."""

from .backends import ExportBackend
from .hooks import (
    compress_weights_openvino_int8_sym,
)
from .mixin_policy import ExportablePolicyMixin


def get_available_backends() -> list[str]:
    """Get list of available export backends.

    Returns:
        List of backend names as strings.

    Examples:
        >>> from physicalai.export import get_available_backends
        >>> backends = get_available_backends()
        >>> print(backends)
        ['onnx', 'openvino', 'torch', 'executorch']
    """
    return [backend.value for backend in ExportBackend]


__all__ = [
    "ExportBackend",
    "ExportablePolicyMixin",
    "compress_weights_openvino_int8_sym",
    "get_available_backends",
]

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for different backend runtimes.

Adapters are discovered through :class:`InferenceBackendRegistry`.  This
package ships the ``onnx`` and ``openvino`` backends only — they
self-register on import below.

Additional backends are contributed by other distributions sharing the
``physicalai`` namespace via the ``physicalai.inference.adapters``
:mod:`importlib.metadata` entry-point group.  Each such entry point names
a callable ``register(registry)`` that populates the shared
:data:`backend_registry`.  Lightweight registration entry points (no heavy
imports) let third parties expose their backends — and in particular
their file extensions for auto-detection — without forcing this package
to know about them.

Use :func:`get_adapter` to obtain an adapter instance for a given backend.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from pkgutil import extend_path
from typing import Any

# Allow `physicalai.inference.adapters` to be split across multiple
# distributions sharing the `physicalai` namespace.
__path__ = extend_path(__path__, __name__)

from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.registry import (
    InferenceBackendRegistry,
    backend_registry,
)

# Eagerly import core adapters so they self-register.  Their runtime
# dependencies (onnxruntime, openvino) are part of the `inference` extra.
from physicalai.inference.adapters.openvino import OpenVINOAdapter
from physicalai.inference.adapters.onnx import ONNXAdapter

logger = logging.getLogger(__name__)

#: Entry-point group used by other distributions to contribute adapters.
#:
#: Each entry point must point at a callable with the signature
#: ``register(registry: InferenceBackendRegistry) -> None``.  The callable
#: should populate *registry* (typically via
#: :meth:`InferenceBackendRegistry.register_lazy_module`) without
#: triggering heavy imports.
ENTRY_POINT_GROUP = "physicalai.inference.adapters"


def _load_external_adapters() -> None:
    """Discover and run third-party adapter registrations.

    Imports every entry point in :data:`ENTRY_POINT_GROUP` and invokes it
    with the shared :data:`backend_registry`.  Failures are logged and
    swallowed so a single broken plugin cannot prevent the rest of the
    inference stack from importing.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            register_fn = ep.load()
        except Exception:
            logger.exception("Failed to load adapter provider entry point %r", ep.name)
            continue
        try:
            register_fn(backend_registry)
        except Exception:
            logger.exception("Adapter provider %r raised during registration", ep.name)


_load_external_adapters()


__all__ = [
    "InferenceBackendRegistry",
    "ONNXAdapter",
    "OpenVINOAdapter",
    "RuntimeAdapter",
    "backend_registry",
    "get_adapter",
]


def get_adapter(backend: str, **kwargs: Any) -> RuntimeAdapter:  # noqa: ANN401
    """Instantiate the adapter registered for *backend*.

    Args:
        backend: Backend identifier as a string (e.g. ``"onnx"``,
            ``"openvino"``).  Any object whose ``str()`` value matches a
            registered backend is accepted (e.g. a
            :class:`enum.StrEnum` member).
        **kwargs: Forwarded to the adapter constructor (e.g. ``device``).

    Returns:
        A ready-to-use :class:`RuntimeAdapter` instance.

    Raises:
        ValueError: If no adapter is registered for *backend*.
        ImportError: If the adapter's optional runtime dependency is not
            installed (raised by the lazy module import inside the
            registry).

    Examples:
        >>> adapter = get_adapter("openvino", device="CPU")
        >>> adapter = get_adapter("onnx")
    """
    name = str(backend)
    adapter_cls = backend_registry.get_class(name)
    return adapter_cls(**kwargs)

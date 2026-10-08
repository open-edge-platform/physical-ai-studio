# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Cosmos 3 policy public entry points."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Cosmos3Config
    from .model import Cosmos3Model
    from .policy import Cosmos3
    from .preprocessor import Cosmos3Preprocessor

_EXPORTS = {
    "Cosmos3": (".policy", "Cosmos3"),
    "Cosmos3Config": (".config", "Cosmos3Config"),
    "Cosmos3Model": (".model", "Cosmos3Model"),
    "Cosmos3Preprocessor": (".preprocessor", "Cosmos3Preprocessor"),
}

__all__ = ["Cosmos3", "Cosmos3Config", "Cosmos3Model", "Cosmos3Preprocessor"]


def __getattr__(name: str) -> Any:  # noqa: ANN401  # PEP 562 module exports are dynamic.
    """Load a Cosmos3 implementation export only when a caller requests it.

    Import errors from the requested Cosmos3 module are propagated.

    Returns:
        The requested Cosmos3 export.

    Raises:
        AttributeError: If the requested name is not a public Cosmos3 export.
    """
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from None

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy Cosmos3 exports in module introspection.

    Returns:
        Names available from this module, including lazy Cosmos3 exports.
    """
    return sorted(set(globals()) | set(_EXPORTS))

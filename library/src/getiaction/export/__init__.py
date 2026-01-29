# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export module - requires getiaction[torch]."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .types import ExportBackend

if TYPE_CHECKING:
    from .mixin_export import Export

__all__ = ["Export", "ExportBackend"]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "Export":
        from .mixin_export import Export  # noqa: PLC0415

        return Export

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return __all__

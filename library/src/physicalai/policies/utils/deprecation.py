# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared deprecation warnings for legacy public APIs."""

from __future__ import annotations

import warnings


def deprecate(name: str, reason: str) -> None:
    """Emit a :class:`DeprecationWarning` for a legacy public API."""
    warnings.warn(
        f"{name} is deprecated and will be removed in a future release; {reason}",
        DeprecationWarning,
        stacklevel=3,
    )

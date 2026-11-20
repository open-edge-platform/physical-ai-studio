# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lazy discovery of engine classes from installed entry points."""

from __future__ import annotations

import re
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from . import Engine

ENTRY_POINT_GROUP = "physicalai.engines"
_ENGINE_NAME_RE = re.compile(r"[a-zA-Z0-9_-]{1,64}")


def _entry_points() -> list[EntryPoint]:
    """Return engine entry points without importing their target modules."""
    return list(entry_points(group=ENTRY_POINT_GROUP))


def names() -> list[str]:
    """List installed engine identifiers without loading any engine code.

    Returns:
        Installed engine identifiers in sorted order.
    """
    return sorted(_entries_by_name())


def get_engine(name: str, engine_base: type[object]) -> type[Engine]:
    """Load and validate the class registered for *name*.

    Returns:
        The class registered for *name*.

    Raises:
        ValueError: If the name is unknown, duplicated, or invalid.
        TypeError: If the entry point does not resolve to an engine subclass.
    """
    entries = _entries_by_name()
    try:
        entry_point = entries[name]
    except KeyError:
        available = ", ".join(entries) or "<none>"
        msg = f"Unknown engine {name!r}. Installed engines: {available}."
        raise ValueError(msg) from None

    engine_class = entry_point.load()
    if not isinstance(engine_class, type) or not issubclass(engine_class, engine_base):
        msg = f"Engine entry point {name!r} must resolve to an engine subclass, got {engine_class!r}."
        raise TypeError(msg)
    engine_class = cast("type[Engine]", engine_class)
    if engine_class.framework != name:
        msg = f"Engine class framework {engine_class.framework!r} does not match entry-point name {name!r}."
        raise ValueError(msg)
    return engine_class


def _entries_by_name() -> dict[str, EntryPoint]:
    entries: dict[str, EntryPoint] = {}
    for entry_point in sorted(_entry_points(), key=_entry_point_sort_key):
        name = entry_point.name
        if _ENGINE_NAME_RE.fullmatch(name) is None:
            msg = f"Engine entry-point name must match [a-zA-Z0-9_-]{{1,64}}, got {name!r}."
            raise ValueError(msg)
        if name in entries:
            msg = f"Duplicate engine entry point {name!r}."
            raise ValueError(msg)
        entries[name] = entry_point
    return entries


def _entry_point_sort_key(entry_point: EntryPoint) -> tuple[str, str, str]:
    distribution = getattr(entry_point, "dist", None)
    distribution_name = getattr(distribution, "name", "") or ""
    value = getattr(entry_point, "value", "") or ""
    return entry_point.name, distribution_name, value


__all__ = ["ENTRY_POINT_GROUP", "get_engine", "names"]

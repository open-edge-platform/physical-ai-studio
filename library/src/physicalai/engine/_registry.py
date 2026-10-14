# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Framework registration and entry-point discovery."""

from __future__ import annotations

import importlib.metadata
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.metadata import EntryPoint

    from ._types import Framework

ENTRY_POINT_GROUP = "physicalai.engine.frameworks"
_FRAMEWORK_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
logger = logging.getLogger(__name__)


class DuplicateFrameworkError(ValueError):
    """Raised when a framework name is already registered."""


class FrameworkRegistry:
    """Registry of frameworks available to :class:`physicalai.engine.Engine`."""

    def __init__(self) -> None:
        self._frameworks: dict[str, Framework] = {}

    @classmethod
    def discover(cls, entry_points: Iterable[EntryPoint] | None = None) -> FrameworkRegistry:
        """Load registration functions from installed framework entry points.

        Entry points expose a ``register(registry)`` callable. Failures to load
        or register an optional plugin are logged and isolated; duplicate
        framework names remain fatal so discovery never silently chooses a
        winner. Each plugin's registrations are rolled back if it fails.

        Args:
            entry_points: Optional entry points for an isolated registry.
                Defaults to the installed ``physicalai.engine.frameworks`` group.

        Returns:
            Registry populated by successfully loaded plugins.

        Raises:
            DuplicateFrameworkError: If two plugins register the same name.
        """
        registry = cls()
        discovered = (
            importlib.metadata.entry_points(group=ENTRY_POINT_GROUP) if entry_points is None else entry_points
        )
        for entry_point in sorted(discovered, key=_entry_point_sort_key):
            snapshot = registry._frameworks.copy()
            try:
                register = entry_point.load()
            except Exception as exc:  # noqa: BLE001  # Optional plugin loads are isolated discovery boundaries.
                logger.warning(
                    "Skipping framework entry point %r during loading after %s.",
                    entry_point.name,
                    type(exc).__name__,
                )
                continue
            if not callable(register):
                logger.warning(
                    "Skipping framework entry point %r because it did not load a register(registry) callable.",
                    entry_point.name,
                )
                continue
            try:
                register(registry)
            except DuplicateFrameworkError:
                registry._frameworks = snapshot
                raise
            except Exception as exc:  # noqa: BLE001  # Optional plugin registration failures must be isolated.
                registry._frameworks = snapshot
                logger.warning(
                    "Skipping framework entry point %r during registration after %s.",
                    entry_point.name,
                    type(exc).__name__,
                )
        return registry

    def register(self, framework: Framework) -> None:
        """Register a framework, rejecting invalid and duplicate names.

        Args:
            framework: Framework implementation to register.

        Raises:
            TypeError: If the object does not implement the framework contract.
            ValueError: If its name is invalid.
            DuplicateFrameworkError: If its name is already registered.
        """
        name = getattr(framework, "name", None)
        if not isinstance(name, str) or _FRAMEWORK_NAME_RE.fullmatch(name) is None:
            msg = f"Framework name must match [a-zA-Z0-9_-]{{1,64}}, got {name!r}."
            raise ValueError(msg)
        missing_methods = [
            method_name
            for method_name in ("accepts", "train", "export")
            if not callable(getattr(framework, method_name, None))
        ]
        if missing_methods:
            msg = f"Framework {name!r} is missing callable methods: {', '.join(missing_methods)}."
            raise TypeError(msg)
        if name in self._frameworks:
            msg = f"Duplicate framework registration: {name!r}."
            raise DuplicateFrameworkError(msg)
        self._frameworks[name] = framework

    def get(self, name: str) -> Framework:
        """Return a registered framework by name.

        Args:
            name: Framework identifier.

        Raises:
            ValueError: If no framework has that name.
        """
        try:
            return self._frameworks[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            msg = f"Unknown framework {name!r}. Registered frameworks: {available}."
            raise ValueError(msg) from exc

    def names(self) -> list[str]:
        """Return registered framework identifiers in registration order."""
        return list(self._frameworks)

    def for_config(self, config: object) -> Framework:
        """Find the unique framework that accepts a native config object.

        Args:
            config: Framework-native configuration object.

        Returns:
            The only framework whose ``accepts()`` method returns ``True``.

        Raises:
            ValueError: If no framework accepts the object or multiple
                frameworks accept it.
        """
        matches = [framework for framework in self._frameworks.values() if framework.accepts(config)]
        if not matches:
            available = ", ".join(self.names()) or "<none>"
            msg = f"No registered framework accepts config type {type(config).__name__}. Available: {available}."
            raise ValueError(msg)
        if len(matches) > 1:
            names = ", ".join(framework.name for framework in matches)
            msg = f"Config is ambiguous: frameworks {names} accept it; pass framework= explicitly."
            raise ValueError(msg)
        return matches[0]


def _entry_point_sort_key(entry_point: EntryPoint) -> tuple[str, str, str]:
    distribution = getattr(entry_point, "dist", None)
    distribution_name = getattr(distribution, "name", "") or ""
    value = getattr(entry_point, "value", "") or ""
    return entry_point.name, distribution_name, value

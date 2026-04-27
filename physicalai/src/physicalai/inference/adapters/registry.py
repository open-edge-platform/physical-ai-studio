# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference backend registry.

The :class:`InferenceBackendRegistry` is a process-wide map from a backend
name (e.g. ``"onnx"``, ``"openvino"``, ``"torch"``, ``"executorch"``) to the
:class:`~physicalai.inference.adapters.base.RuntimeAdapter` subclass that
implements it together with the file extensions that identify model
artifacts produced for that backend.

Adapters self-register on import via the
:meth:`InferenceBackendRegistry.register` decorator so that adapters
contributed by different distributions (e.g. core ``physicalai`` provides
ONNX and OpenVINO; ``physicalai-train`` provides Torch and ExecuTorch) all
end up in the same shared registry.

To keep heavy dependencies opt-in, the registry also supports
*lazy module registration*: a backend name plus its extensions can be
associated with a module import path that, when imported, is expected to
register a concrete adapter class.  The module is imported only when the
backend is actually requested via :meth:`get_class`.  Crucially, the
extensions are available immediately, so callers can probe export
directories without paying the import cost.

Examples:
    Self-registering an adapter from its module::

        from physicalai.inference.adapters.registry import backend_registry
        from physicalai.inference.adapters.base import RuntimeAdapter

        @backend_registry.register("onnx", extensions=(".onnx",))
        class ONNXAdapter(RuntimeAdapter):
            ...

    Pre-declaring a backend whose adapter lives in another distribution::

        backend_registry.register_lazy_module(
            "torch",
            "physicalai.inference.adapters.torch",
            extensions=(".ckpt", ".pt"),
        )
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from physicalai.inference.adapters.base import RuntimeAdapter

AdapterT = TypeVar("AdapterT", bound="RuntimeAdapter")


class InferenceBackendRegistry:
    """Registry of backend name → :class:`RuntimeAdapter` subclass + extensions.

    Each entry tracks the adapter class (eager or lazily resolvable) and the
    file extensions that identify model artifacts for that backend.  The
    extension list lets callers detect a backend from an export directory
    without importing optional adapter dependencies.

    Supports two registration modes:

    * **Eager** — :meth:`register` is used as a decorator on the adapter
      class itself and inserts it into the registry as soon as the adapter
      module is imported.
    * **Lazy** — :meth:`register_lazy_module` pre-declares a backend name,
      its extensions, and the module that, when imported, will eagerly
      register the adapter class for that name.

    The registry is intended to be used as the module-level singleton
    :data:`backend_registry`.  Iteration order reflects registration
    order, so the first registered backend whose extension matches an
    on-disk artifact wins during file-based auto-detection.

    Examples:
        >>> from physicalai.inference.adapters.registry import backend_registry
        >>> "onnx" in backend_registry
        True
        >>> backend_registry.extensions_of("onnx")
        ('.onnx',)
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._classes: dict[str, type[RuntimeAdapter]] = {}
        self._lazy: dict[str, str] = {}
        self._extensions: dict[str, tuple[str, ...]] = {}

    def register(
        self,
        name: str,
        *,
        extensions: Iterable[str],
    ) -> Callable[[type[AdapterT]], type[AdapterT]]:
        """Return a decorator that registers an adapter class for *name*.

        Args:
            name: Backend identifier, e.g. ``"onnx"``.
            extensions: File extensions (with leading dot) that identify
                exported artifacts for this backend, in priority order.

        Returns:
            A class decorator that records the class in the registry and
            returns it unchanged.
        """
        normalized = self._normalize_extensions(extensions)

        def _wrap(cls: type[AdapterT]) -> type[AdapterT]:
            self._classes[name] = cls
            self._lazy.pop(name, None)
            self._extensions.setdefault(name, normalized)
            return cls

        return _wrap

    def register_lazy_module(
        self,
        name: str,
        module_path: str,
        *,
        extensions: Iterable[str],
    ) -> None:
        """Pre-declare a backend whose adapter is registered on first use.

        The first call to :meth:`get_class` for *name* will import
        *module_path*; importing the module is expected to call
        :meth:`register` (typically via the decorator) for *name*.
        Eager registrations always win over a lazy declaration with the
        same name.

        Args:
            name: Backend identifier, e.g. ``"torch"``.
            module_path: Fully-qualified module path to import on first use.
            extensions: File extensions for this backend, in priority order.
                Available immediately so callers can detect the backend
                from on-disk artifacts without triggering the import.
        """
        if name not in self._classes:
            self._lazy[name] = module_path
        self._extensions.setdefault(name, self._normalize_extensions(extensions))

    def get_class(self, name: str) -> type[RuntimeAdapter]:
        """Return the adapter class registered for *name*.

        If only a lazy module registration exists, imports the module
        (which is expected to register the adapter) and then returns the
        resolved class.

        Args:
            name: Backend identifier.

        Returns:
            The :class:`RuntimeAdapter` subclass for the requested backend.

        Raises:
            ValueError: If *name* is unknown.
            ImportError: Re-raised from the lazy import if the module
                cannot be imported (e.g. missing optional dependency).
            RuntimeError: If the lazy module imported successfully but
                failed to register an adapter for *name*.
        """
        if name in self._classes:
            return self._classes[name]

        if name in self._lazy:
            module_path = self._lazy[name]
            importlib.import_module(module_path)
            if name in self._classes:
                return self._classes[name]
            msg = (
                f"Module {module_path!r} was imported for backend {name!r} "
                "but did not register an adapter."
            )
            raise RuntimeError(msg)

        available = ", ".join(self.names()) or "<none>"
        msg = f"No adapter registered for backend {name!r}. Registered backends: {available}"
        raise ValueError(msg)

    def extensions_of(self, name: str) -> tuple[str, ...]:
        """Return the file extensions registered for backend *name*.

        Args:
            name: Backend identifier.

        Returns:
            Tuple of extensions (with leading dot), in priority order.

        Raises:
            ValueError: If *name* is unknown.
        """
        if name not in self._extensions:
            available = ", ".join(self.names()) or "<none>"
            msg = f"No extensions registered for backend {name!r}. Registered backends: {available}"
            raise ValueError(msg)
        return self._extensions[name]

    def detect_by_extension(self, extension: str) -> str | None:
        """Return the backend name that owns *extension*, if any.

        Iteration follows registration order, so earlier-registered
        backends take priority when extensions overlap.

        Args:
            extension: File extension to look up (with leading dot,
                case-insensitive).

        Returns:
            Backend name, or ``None`` if no backend claims the extension.
        """
        ext = extension.lower()
        for backend, exts in self._extensions.items():
            if ext in exts:
                return backend
        return None

    def names(self) -> list[str]:
        """Return all registered backend names (eager and lazy) in registration order."""
        seen: dict[str, None] = {}
        for source in (self._classes, self._lazy, self._extensions):
            for name in source:
                seen.setdefault(name)
        return list(seen)

    def __contains__(self, name: str) -> bool:
        """Return whether *name* is registered (eager or lazy)."""
        return name in self._classes or name in self._lazy

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{type(self).__name__}("
            f"eager={list(self._classes)!r}, lazy={list(self._lazy)!r})"
        )

    @staticmethod
    def _normalize_extensions(extensions: Iterable[str]) -> tuple[str, ...]:
        """Lower-case and validate file extensions.

        Args:
            extensions: Iterable of extension strings.

        Returns:
            Tuple of normalized extensions in input order.

        Raises:
            ValueError: If any extension is empty or missing a leading dot.
        """
        normalized: list[str] = []
        for ext in extensions:
            if not ext or not ext.startswith("."):
                msg = f"Extension must start with '.': got {ext!r}"
                raise ValueError(msg)
            normalized.append(ext.lower())
        return tuple(normalized)


#: Process-wide singleton used by :func:`get_adapter` and adapter modules.
backend_registry = InferenceBackendRegistry()

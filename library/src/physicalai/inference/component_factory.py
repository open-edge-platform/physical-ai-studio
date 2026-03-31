# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Component registry and factory for dynamic instantiation.

The :class:`ComponentRegistry` maps short names (e.g. ``"single_pass"``)
to fully-qualified class paths so that manifests can use concise
identifiers instead of full dotted paths.  The :func:`instantiate_component`
factory resolves a :class:`~physicalai.inference.manifest.ComponentSpec`
to an object instance, consulting the registry when the ``class_path``
contains no dot (i.e. is a short name).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from physicalai.inference.manifest import ComponentSpec


class ComponentRegistry:
    """Name → class_path registry for dynamically instantiated components.

    Built-in entries are registered at module load time.  Domain layers
    can register additional entries via :meth:`register`.

    Examples:
        >>> registry = ComponentRegistry()
        >>> registry.register("my_runner", "myapp.runners.MyRunner")
        >>> registry.resolve("my_runner")
        'myapp.runners.MyRunner'
        >>> registry.resolve("myapp.runners.MyRunner")  # passthrough
        'myapp.runners.MyRunner'
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._entries: dict[str, str] = {}

    def register(self, name: str, class_path: str) -> None:
        """Register a short name → class_path mapping.

        Args:
            name: Short identifier (e.g. ``"single_pass"``).
            class_path: Fully-qualified class path.
        """
        self._entries[name] = class_path

    def resolve(self, name_or_path: str) -> str:
        """Resolve a short name to a class path, or pass through if already qualified.

        Args:
            name_or_path: Either a registered short name or a full class path.

        Returns:
            Fully-qualified class path.
        """
        return self._entries.get(name_or_path, name_or_path)

    def get_class(self, name_or_path: str) -> type:
        """Resolve a name to a class path, import the module, and return the class.

        Args:
            name_or_path: Either a registered short name or a full class path.

        Returns:
            The resolved class object.
        """
        class_path = self.resolve(name_or_path)
        module_path, class_name = class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def entries(self) -> dict[str, str]:
        """Return a copy of all registered entries.

        Returns:
            Dict mapping short names to class paths.
        """
        return dict(self._entries)

    def __contains__(self, name: str) -> bool:
        """Check if *name* is a registered short name.

        Returns:
            True if *name* is registered.
        """
        return name in self._entries

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"ComponentRegistry({self._entries!r})"


component_registry = ComponentRegistry()

# Runners
component_registry.register("single_pass", "physicalai.inference.runners.SinglePass")
component_registry.register("action_chunking", "physicalai.inference.runners.ActionChunking")

# Postprocessors
component_registry.register("smolvla_resize", "physicalai.inference.preprocessors.smolvla.ResizeSmolVLA")


def instantiate_component(
    spec: ComponentSpec,
    *,
    registry: ComponentRegistry | None = None,
) -> object:
    """Import the class described by *spec* and return a live instance.

    If ``spec.class_path`` is a registered short name in the *registry*,
    it is resolved to the full class path before import.

    Nested ``ComponentSpec`` dicts in ``init_args`` are instantiated
    recursively.

    Args:
        spec: Component descriptor with class_path and init_args.
        registry: Optional registry for short-name resolution.
            Defaults to :data:`component_registry`.

    Returns:
        An instance of the class specified by spec.class_path.
    """
    reg = registry or component_registry
    class_path = reg.resolve(spec.class_path)

    module_path, class_name = class_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_path)
    cls_obj = getattr(module, class_name)

    resolved_args: dict[str, object] = {}
    for key, value in spec.init_args.items():
        if isinstance(value, dict) and "class_path" in value:
            resolved_args[key] = instantiate_component(
                type(spec).model_validate(value),
                registry=reg,
            )
        else:
            resolved_args[key] = value

    return cls_obj(**resolved_args)

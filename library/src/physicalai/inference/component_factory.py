# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Component registry and factory for dynamic instantiation.

The :class:`ComponentRegistry` maps short names (e.g. ``"single_pass"``)
to fully-qualified class paths so that manifests can use concise
identifiers instead of full dotted paths.  The :func:`instantiate_component`
factory resolves a :class:`~physicalai.inference.manifest.ComponentSpec`
to an object instance, supporting both ``type`` + flat params and
``class_path`` + ``init_args`` resolution modes.
"""

from __future__ import annotations

import importlib
from pathlib import Path
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

# Preprocessors
component_registry.register("normalize", "physicalai.inference.preprocessors.StatsNormalizer")
component_registry.register("smolvla_resize", "physicalai.inference.preprocessors.ResizeSmolVLA")
component_registry.register("new_line", "physicalai.inference.preprocessors.NewLinePreprocessor")
component_registry.register("hf_tokenizer", "physicalai.inference.preprocessors.HFTokenizer")

# Postprocessors
component_registry.register("denormalize", "physicalai.inference.postprocessors.StatsDenormalizer")


def resolve_artifact(spec: ComponentSpec, export_dir: Path) -> ComponentSpec:
    """Resolve relative ``artifact`` paths to absolute paths.

    For type-based specs, resolves a relative ``artifact`` flat
    param to an absolute path.  For class_path-based specs,
    resolves a relative ``artifact`` in ``init_args``.

    Args:
        spec: Component descriptor that may contain a relative artifact path.
        export_dir: Base directory for resolving relative paths.

    Returns:
        The spec with resolved artifact path, or the original spec
        unchanged if no resolution is needed.
    """
    flat = spec.flat_params
    if "artifact" in flat and not Path(flat["artifact"]).is_absolute():
        resolved_path = str(export_dir / flat["artifact"])
        new_params = {**flat, "artifact": resolved_path}
        return type(spec).model_validate({
            "type": spec.type,
            **new_params,
        })

    if spec.class_path and "artifact" in spec.init_args:
        artifact = spec.init_args["artifact"]
        if not Path(artifact).is_absolute():
            new_init_args = {**spec.init_args, "artifact": str(export_dir / artifact)}
            return type(spec).model_validate({
                "class_path": spec.class_path,
                "init_args": new_init_args,
            })

    return spec


def _import_class(class_path: str) -> type:
    """Import and return a class from a fully-qualified dotted path.

    Returns:
        The imported class object.
    """
    module_path, class_name = class_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_component(
    spec: ComponentSpec,
    *,
    registry: ComponentRegistry | None = None,
) -> object:
    """Import the class described by *spec* and return a live instance.

    Supports two resolution modes:

    1. **class_path + init_args** — ``spec.class_path`` is resolved
       (via registry if it's a short name) and the class is
       instantiated with ``spec.init_args``.
    2. **type + flat params** — ``spec.type`` is resolved via the
       registry to a class path, and ``spec.flat_params`` are passed
       as keyword arguments.

    ``class_path`` takes precedence when both are present.

    Nested ``ComponentSpec`` dicts in ``init_args`` are instantiated
    recursively.

    Args:
        spec: Component descriptor with type or class_path.
        registry: Optional registry for short-name resolution.
            Defaults to :data:`component_registry`.

    Returns:
        An instance of the resolved class.
    """
    reg = registry or component_registry

    if spec.class_path:
        resolved_path = reg.resolve(spec.class_path)
        cls_obj = _import_class(resolved_path)

        resolved_args: dict[str, object] = {}
        for key, value in spec.init_args.items():
            if isinstance(value, dict) and ("class_path" in value or "type" in value):
                resolved_args[key] = instantiate_component(
                    type(spec).model_validate(value),
                    registry=reg,
                )
            else:
                resolved_args[key] = value

        return cls_obj(**resolved_args)

    resolved_path = reg.resolve(spec.type)
    cls_obj = _import_class(resolved_path)

    resolved_params: dict[str, object] = {}
    for key, value in spec.flat_params.items():
        if isinstance(value, dict) and ("class_path" in value or "type" in value):
            resolved_params[key] = instantiate_component(
                type(spec).model_validate(value),
                registry=reg,
            )
        else:
            resolved_params[key] = value

    return cls_obj(**resolved_params)

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration instantiation for creating objects from various configuration formats.

Supports YAML/JSON files, dicts, dataclasses, and Pydantic models with class_path pattern.
This module provides the core instantiation functionality.
"""

import dataclasses
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from typing import Any


def _import_class(class_path: str) -> type:
    """Import a class from a module path.

    Args:
        class_path: Full path to class (e.g., 'torch.nn.Linear')

    Returns:
        The imported class

    Raises:
        ImportError: If module or class cannot be imported
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)  # nosemgrep
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Cannot import '{class_path}': {e}"
        raise ImportError(msg) from e


def _instantiate_recursive(value: "Any") -> "Any":  # noqa: ANN401
    """Walk a value and instantiate any nested ``{class_path, init_args}`` dicts.

    Recurses uniformly through dicts, lists, and tuples so nested ``class_path`` configs
    are resolved at any depth, regardless of container type.

    Args:
        value: Arbitrary value that may contain nested ``class_path`` dicts.

    Returns:
        The value with nested ``class_path`` configs replaced by instances.
    """
    if isinstance(value, dict):
        if "class_path" in value:
            return instantiate_obj_from_dict(value)
        return {k: _instantiate_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_instantiate_recursive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_instantiate_recursive(item) for item in value)
    return value


def instantiate_obj_from_dict(
    config: dict[str, "Any"],
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from a configuration dictionary.

    Supports two patterns, auto-detected:

    1. **jsonargparse pattern**: ``{"class_path": "...", "init_args": {...}}``. The class
       is imported from ``class_path`` and ``init_args`` are passed as keyword arguments.
    2. **Direct args** (when ``target_cls`` is provided and ``class_path`` is absent):
       the entire ``config`` dict is treated as keyword arguments to ``target_cls``.

    If both ``class_path`` and ``target_cls`` are supplied, ``class_path`` wins so YAML
    can override the concrete class (subclass dispatch).

    In all cases, nested values are recursively scanned for ``{class_path, init_args}``
    dicts and instantiated.

    Args:
        config: Configuration dictionary.
        key: Optional key to extract a sub-config from ``config`` before instantiating.
        target_cls: Optional fallback class used when ``config`` has no ``class_path``.

    Returns:
        Instantiated object.

    Raises:
        ValueError: If ``key`` is given but missing from ``config``, or if neither
            ``class_path`` nor ``target_cls`` is available.
    """
    if key is not None:
        if key not in config:
            msg = f"Configuration must contain '{key}' key. Got keys: {list(config.keys())}"
            raise ValueError(msg)
        config = config[key]

    if "class_path" in config:
        cls = _import_class(config["class_path"])
        init_args = config.get("init_args", {})
    elif target_cls is not None:
        cls = target_cls
        init_args = config
    else:
        msg = (
            "Configuration must contain 'class_path' for instantiation, "
            f"or pass target_cls explicitly. Got keys: {list(config.keys())}"
        )
        raise ValueError(msg)

    if not isinstance(init_args, dict):
        return cls(init_args)

    instantiated_args = {k: _instantiate_recursive(v) for k, v in init_args.items()}

    # Special handling for classes like nn.Sequential that take *args.
    if "args" in instantiated_args:
        args = instantiated_args.pop("args")
        return cls(*args, **instantiated_args)
    return cls(**instantiated_args)


def instantiate_obj_from_pydantic(
    config: BaseModel,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from a Pydantic model.

    Args:
        config: Pydantic model instance.
        key: Optional key to extract a sub-config before instantiation.
        target_cls: Optional fallback class when ``config`` has no ``class_path``.

    Returns:
        Instantiated object.
    """
    return instantiate_obj_from_dict(config.model_dump(), key=key, target_cls=target_cls)


def instantiate_obj_from_dataclass(
    config: object,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from a dataclass instance.

    Args:
        config: Dataclass instance.
        key: Optional key to extract a sub-config before instantiation.
        target_cls: Optional fallback class when ``config`` has no ``class_path``.

    Returns:
        Instantiated object.

    Raises:
        TypeError: If config is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        msg = f"Expected dataclass instance, got {type(config)}"
        raise TypeError(msg)

    return instantiate_obj_from_dict(dataclasses.asdict(config), key=key, target_cls=target_cls)


def instantiate_obj_from_file(
    file_path: str | Path,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from a YAML/JSON configuration file.

    Args:
        file_path: Path to configuration file.
        key: Optional key to extract a sub-config before instantiation.
        target_cls: Optional fallback class when the file has no ``class_path``.

    Returns:
        Instantiated object.
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return instantiate_obj_from_dict(config, key=key, target_cls=target_cls)


def instantiate_obj(
    config: dict[str, "Any"] | BaseModel | object | str | Path,
    *,
    key: str | None = None,
    target_cls: type | None = None,
) -> object:
    """Instantiate an object from various configuration formats.

    Auto-routes by input type to the appropriate ``instantiate_obj_from_*`` helper.
    Supports both jsonargparse pattern (``{class_path, init_args}``) and direct-args
    pattern (when ``target_cls`` is provided).

    Args:
        config: Configuration as dict, Pydantic model, dataclass, or file path.
        key: Optional sub-config key to extract before instantiation.
        target_cls: Optional fallback class for direct-args mode.

    Returns:
        Instantiated object.

    Raises:
        TypeError: If configuration type is unsupported.

    Examples:
        ```python
        # Class_path pattern from file
        optimizer = instantiate_obj("config.yaml", key="optimizer")

        # Direct args via target_cls
        layer = instantiate_obj({"in_features": 128, "out_features": 64}, target_cls=nn.Linear)
        ```
    """
    if isinstance(config, (str, Path)):
        return instantiate_obj_from_file(config, key=key, target_cls=target_cls)
    if isinstance(config, BaseModel):
        return instantiate_obj_from_pydantic(config, key=key, target_cls=target_cls)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return instantiate_obj_from_dataclass(config, key=key, target_cls=target_cls)
    if isinstance(config, dict):
        return instantiate_obj_from_dict(config, key=key, target_cls=target_cls)

    msg = f"Unsupported configuration type: {type(config)}. Expected dict, file path, Pydantic model, or dataclass."
    raise TypeError(msg)

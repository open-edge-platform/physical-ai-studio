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


def instantiate_obj_from_dict(config: dict[str, "Any"], *, key: str | None = None) -> object:
    """Instantiate an object from a configuration dictionary using jsonargparse pattern.

    Args:
        config: Config dict with class_path and optional init_args, or dict containing sections
        key: Optional key to extract from config dict before instantiation

    Returns:
        Instantiated object

    Raises:
        ValueError: If config dict does not contain 'class_path' or specified key
    """
    # If key is specified, extract that section from the config
    if key is not None:
        if key not in config:
            msg = f"Configuration must contain '{key}' key. Got keys: {list(config.keys())}"
            raise ValueError(msg)
        config = config[key]

    if "class_path" not in config:
        msg = f"Configuration must contain 'class_path' key for instantiation. Got keys: {list(config.keys())}"
        raise ValueError(msg)

    class_path = config["class_path"]
    init_args = config.get("init_args", {})

    # Recursively instantiate nested configs
    if isinstance(init_args, dict):
        instantiated_args = {}
        for arg_key, value in init_args.items():
            if isinstance(value, dict) and "class_path" in value:
                instantiated_args[arg_key] = instantiate_obj(value)
            elif isinstance(value, dict):
                instantiated_args[arg_key] = {
                    k: instantiate_obj(v) if isinstance(v, dict) and "class_path" in v else v for k, v in value.items()
                }
            elif isinstance(value, list):
                instantiated_args[arg_key] = [
                    instantiate_obj(item) if isinstance(item, dict) and "class_path" in item else item for item in value
                ]
            else:
                instantiated_args[arg_key] = value
        init_args = instantiated_args

    # Import and instantiate the class
    cls = _import_class(class_path)

    # Handle special case for classes that take positional args
    if isinstance(init_args, dict):
        if "args" in init_args:
            # Special handling for classes like nn.Sequential that take *args
            args = init_args.pop("args")
            return cls(*args, **init_args)
        return cls(**init_args)
    return cls(init_args)


def instantiate_obj_from_pydantic(config: BaseModel, *, key: str | None = None) -> object:
    """Instantiate an object from a Pydantic model.

    Args:
        config: Pydantic model instance
        key: Optional key to extract from config before instantiation

    Returns:
        Instantiated object
    """
    config_dict = config.model_dump()
    return instantiate_obj_from_dict(config_dict, key=key)


def instantiate_obj_from_dataclass(config: object, *, key: str | None = None) -> object:
    """Instantiate an object from a dataclass instance.

    Args:
        config: Dataclass instance
        key: Optional key to extract from config before instantiation

    Returns:
        Instantiated object

    Raises:
        TypeError: If config is not a dataclass instance
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        msg = f"Expected dataclass instance, got {type(config)}"
        raise TypeError(msg)

    config_dict = dataclasses.asdict(config)
    return instantiate_obj_from_dict(config_dict, key=key)


def instantiate_obj_from_file(file_path: str | Path, *, key: str | None = None) -> object:
    """Instantiate an object from a YAML/JSON configuration file.

    Args:
        file_path: Path to configuration file
        key: Optional key to extract from config before instantiation

    Returns:
        Instantiated object
    """
    with Path(file_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return instantiate_obj_from_dict(config, key=key)


def instantiate_obj(config: dict[str, "Any"] | BaseModel | object | str | Path, *, key: str | None = None) -> object:
    """Instantiate an object from various configuration formats using jsonargparse pattern.

    This is the main entry point that automatically routes to the appropriate
    type-specific instantiation function based on the input type. It expects
    configurations to follow the jsonargparse pattern with 'class_path'.

    Args:
        config: Configuration in any supported format (dict, Pydantic model, dataclass, or file path)
        key: Optional key to extract from config before instantiation (useful for multi-section configs)

    Returns:
        Instantiated object

    Raises:
        TypeError: If configuration type is unsupported

    Examples:
        ```python
        # From file - instantiate specific section
        optimizer = instantiate_obj("config.yaml", key="optimizer")

        # From dictionary - instantiate specific section
        config = {"model": {"class_path": "torch.nn.Linear", "init_args": {"in_features": 128}}}
        layer = instantiate_obj(config, key="model")

        # From dictionary - direct instantiation
        config = {"class_path": "torch.nn.Linear", "init_args": {"in_features": 128, "out_features": 64}}
        layer = instantiate_obj(config)
        ```
    """
    # Route to appropriate type-specific function
    if isinstance(config, (str, Path)):
        return instantiate_obj_from_file(config, key=key)
    if isinstance(config, BaseModel):
        return instantiate_obj_from_pydantic(config, key=key)
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return instantiate_obj_from_dataclass(config, key=key)
    if isinstance(config, dict):
        return instantiate_obj_from_dict(config, key=key)

    # For any other type, raise error
    msg = (
        f"Unsupported configuration type: {type(config)}. "
        "Expected dict with 'class_path', file path, Pydantic model, or dataclass."
    )
    raise TypeError(msg)

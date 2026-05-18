# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration mixins for adding from_config functionality to any class.

This module provides a thin sugar layer on top of the generic instantiation
utilities in :mod:`physicalai.config.instantiate` and the dataclass conversion
utilities in :mod:`physicalai.config.serializable`. The mixin and decorator
both delegate to those utilities so there is a single source of truth for
recursion, ``class_path`` dispatch, and dataclass/Pydantic conversion.
"""

import dataclasses
from pathlib import Path
from typing import Any, Self, cast

import yaml
from pydantic import BaseModel

from physicalai.config.instantiate import instantiate_obj_from_dict
from physicalai.config.serializable import dataclass_to_dict


class FromConfig:
    """Mixin that adds configuration-based instantiation helpers to any class.

    Auto-detects two patterns:

    1. **jsonargparse pattern**: ``{"class_path": "...", "init_args": {...}}`` — the
       class encoded in ``class_path`` is instantiated (allows subclass dispatch via
       config files).
    2. **Direct args**: a flat dict of constructor kwargs — instantiates ``cls`` directly.

    Both patterns are handled by :func:`instantiate_obj_from_dict` with ``target_cls=cls``.

    Examples:
        ```python
        class MyModel(FromConfig):
            def __init__(self, hidden_size: int, num_layers: int) -> None:
                self.hidden_size = hidden_size
                self.num_layers = num_layers

        # Direct args
        MyModel.from_dict({"hidden_size": 128, "num_layers": 3})

        # class_path dispatch (could be a subclass)
        MyModel.from_dict({"class_path": "pkg.MyBigModel", "init_args": {...}})

        # YAML / Pydantic / dataclass
        MyModel.from_yaml("config.yaml")
        MyModel.from_pydantic(pyd_cfg)
        MyModel.from_dataclass(dc_cfg)
        MyModel.from_config(any_of_the_above)
        ```
    """

    @classmethod
    def from_yaml(cls, file_path: str | Path, *, key: str | None = None) -> Self:
        """Load configuration from a YAML file and instantiate the class.

        Args:
            file_path: Path to the YAML configuration file.
            key: Optional key to extract a sub-configuration from the file.

        Returns:
            An instance of the class.
        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config, key=key)

    @classmethod
    def from_dict(cls, config: dict[str, Any], *, key: str | None = None) -> Self:
        """Instantiate the class from a configuration dictionary.

        Auto-detects ``class_path``/``init_args`` vs direct kwargs. Nested
        ``class_path`` dicts at any depth are also instantiated.

        Args:
            config: Configuration dictionary.
            key: Optional sub-config key to extract before instantiation.

        Returns:
            An instance of the class.
        """
        return cast("Self", instantiate_obj_from_dict(config, key=key, target_cls=cls))

    @classmethod
    def from_pydantic(
        cls,
        config: BaseModel,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Instantiate the class from a Pydantic model.

        Args:
            config: Pydantic model instance.
            key: Optional sub-config key to extract before instantiation.
            recursive: If ``True``, recursively dump nested Pydantic models to dicts.
                If ``False`` (default), preserves nested Pydantic models as instances
                so they can be passed straight to constructors that expect them.

        Returns:
            An instance of the class.
        """
        if recursive:
            config_dict = config.model_dump()
        else:
            config_dict = {name: getattr(config, name) for name in config.__class__.model_fields}
        return cls.from_dict(config_dict, key=key)

    @classmethod
    def from_dataclass(
        cls,
        config: object,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Instantiate the class from a dataclass instance.

        Args:
            config: Dataclass instance.
            key: Optional sub-config key to extract before instantiation.
            recursive: If ``True``, recursively convert nested dataclasses to dicts.
                If ``False`` (default), preserves nested dataclass instances so they
                can be passed to constructors that expect dataclass arguments.

        Returns:
            An instance of the class.

        Raises:
            TypeError: If ``config`` is not a dataclass instance.
        """
        if not dataclasses.is_dataclass(config) or isinstance(config, type):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)

        config_dict = cast("dict[str, Any]", dataclass_to_dict(config, recursive=recursive))
        return cls.from_dict(config_dict, key=key)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | BaseModel | object | str | Path,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Generic entry point that dispatches on the type of ``config``.

        Args:
            config: Configuration as dict, file path, Pydantic model, or dataclass.
            key: Optional sub-config key to extract before instantiation.
            recursive: For dataclass and Pydantic configs, controls whether nested
                instances are flattened to dicts (``True``) or preserved (``False``,
                default). Ignored for dicts and YAML paths.

        Returns:
            An instance of the class.

        Raises:
            TypeError: If ``config`` is of an unsupported type.
        """
        if isinstance(config, (str, Path)):
            return cls.from_yaml(config, key=key)
        if isinstance(config, BaseModel):
            return cls.from_pydantic(config, key=key, recursive=recursive)
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            return cls.from_dataclass(config, key=key, recursive=recursive)
        if isinstance(config, dict):
            return cls.from_dict(config, key=key)

        msg = f"Unsupported configuration type: {type(config)}. Expected dict, file path, Pydantic model, or dataclass."
        raise TypeError(msg)


def from_config[T: type](cls: T) -> T:
    """Decorate a class with the same config constructors as :class:`FromConfig`.

    Equivalent to inheriting from :class:`FromConfig`, but useful for classes that
    cannot easily change their MRO (e.g. when subclassing third-party classes).

    Args:
        cls: Class to decorate.

    Returns:
        The same class with ``from_yaml``/``from_dict``/``from_pydantic``/
        ``from_dataclass``/``from_config`` classmethods attached.
    """
    for name in ("from_yaml", "from_dict", "from_pydantic", "from_dataclass", "from_config"):
        setattr(cls, name, FromConfig.__dict__[name])
    return cls

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration mixins for adding from_config functionality to any class.

This module provides mixins that add generic configuration loading capabilities
to any class, supporting YAML files, Pydantic models, dataclasses, and dictionaries.
"""

import dataclasses
from pathlib import Path
from typing import Any, Self, cast

import yaml
from pydantic import BaseModel

from physicalai.config.instantiate import instantiate_obj_from_dict


class FromConfig:
    """Enhanced mixin class that provides comprehensive configuration loading functionality.

    This mixin can be inherited by any class to automatically provide
    configuration-based instantiation capabilities. It supports multiple
    configuration formats and automatically detects the appropriate pattern.

    The mixin automatically detects two patterns:
    1. **jsonargparse pattern**: Configuration with 'class_path' and 'init_args' keys
    2. **Direct instantiation**: Configuration that directly maps to constructor arguments

    Examples:
        ```python
        class MyModel(nn.Module, FromConfig):
            def __init__(self, hidden_size: int, num_layers: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers

        # From YAML file (auto-detects pattern)
        model = MyModel.from_yaml("model_config.yaml")

        # From dictionary (jsonargparse pattern - auto-detected)
        config = {
            "class_path": "my.module.MyModel",
            "init_args": {"hidden_size": 128, "num_layers": 3}
        }
        model = MyModel.from_dict(config)

        # From dictionary (direct instantiation - auto-detected)
        config = {"hidden_size": 128, "num_layers": 3}
        model = MyModel.from_dict(config)

        # From Pydantic model (auto-detects pattern)
        class ModelConfig(BaseModel):
            hidden_size: int = 128
            num_layers: int = 3

        model = MyModel.from_pydantic(ModelConfig())

        # From dataclass (auto-detects pattern)
        @dataclass
        class ModelConfig:
            hidden_size: int = 128
            num_layers: int = 3

        model = MyModel.from_dataclass(ModelConfig())

        # Generic method (auto-detects format and pattern)
        model = MyModel.from_config("model_config.yaml")
        model = MyModel.from_config(config_dict)
        model = MyModel.from_config(pydantic_config)
        ```
    """

    @classmethod
    def from_yaml(
        cls,
        file_path: str | Path,
        *,
        key: str | None = None,
    ) -> Self:
        """Load configuration from a YAML file and instantiate the class.

        Automatically detects whether to use jsonargparse pattern (class_path/init_args)
        or direct instantiation based on the configuration structure.

        Args:
            file_path: Path to the YAML configuration file.
            key: Optional key to extract a sub-configuration from the file.

        Returns:
            An instance of the class.

        Examples:
            ```python
            # Auto-detects pattern based on config structure
            model = MyModel.from_yaml("config.yaml")

            # Extract specific section
            model = MyModel.from_yaml("config.yaml", key="model")
            ```
        """
        # Load file and auto-detect pattern
        with Path(file_path).open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config, key=key)

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        *,
        key: str | None = None,
    ) -> Self:
        """Load configuration from a dictionary and instantiate the class.

        Automatically detects whether to use jsonargparse pattern (class_path/init_args)
        or direct instantiation based on the configuration structure.

        Args:
            config: Configuration dictionary.
            key: Optional key to extract a sub-configuration from the dict.

        Returns:
            An instance of the class.

        Raises:
            ValueError: If key is not found in config.

        Examples:
            1. Jsonargparse pattern (auto-detected)
            ```python
            config = {
                "class_path": "my.module.MyModel",
                "init_args": {"hidden_size": 128, "num_layers": 3}
            }
            model = MyModel.from_dict(config)
            ```

            2. Direct instantiation (auto-detected)
            ```python
            config = {"hidden_size": 128, "num_layers": 3}
            model = MyModel.from_dict(config)
            ```
        """
        # Extract key if specified
        if key is not None:
            if key not in config:
                msg = f"Configuration must contain '{key}' key. Got keys: {list(config.keys())}"
                raise ValueError(msg)
            config = config[key]

        # Auto-detect pattern: if both class_path and init_args exist, use jsonargparse pattern
        if "class_path" in config and "init_args" in config:
            instance = instantiate_obj_from_dict(config)
        else:
            instance = cls(**config)

        return cast("Self", instance)

    @classmethod
    def from_pydantic(
        cls,
        config: BaseModel,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Load configuration from a Pydantic model and instantiate the class.

        Automatically detects whether to use jsonargparse pattern (class_path/init_args)
        or direct instantiation based on the configuration structure.

        Args:
            config: Pydantic model instance.
            key: Optional key to extract a sub-configuration from the model.
            recursive: If False (default), preserves nested Pydantic models as instances.
                If True, recursively converts nested Pydantic models to dicts.
                This mirrors the behavior of from_dataclass.

        Returns:
            An instance of the class.

        Examples:
            1. Jsonargparse pattern (auto-detected)
            ```python
            class ModelConfig(BaseModel):
                class_path: str = "my.module.MyModel"
                init_args: dict = {"hidden_size": 128, "num_layers": 3}

            model = MyModel.from_pydantic(ModelConfig())
            ```

            2. Direct instantiation (auto-detected)
            ```python
            class ModelConfig(BaseModel):
                hidden_size: int = 128
                num_layers: int = 3

            model = MyModel.from_pydantic(ModelConfig())
            ```

            3. With recursive control for nested models (mirrors from_dataclass)
            ```python
            # recursive=False: Preserves nested Pydantic models as instances (default)
            # Use when constructor expects Pydantic model instances
            model = MyModel.from_pydantic(config, recursive=False)

            # recursive=True: Converts all nested Pydantic models to dicts
            # Use when constructor expects plain dictionaries
            model = MyModel.from_pydantic(config, recursive=True)
            ```
        """
        # Mirror the dataclass behavior: recursive controls nested object conversion
        if recursive:
            # Convert nested Pydantic models to dicts (like dataclasses.asdict)
            config_dict = config.model_dump()
        else:
            # Preserve nested Pydantic models as instances (like manual field access)
            config_dict = {field_name: getattr(config, field_name) for field_name in config.__class__.model_fields}

        return cls.from_dict(config_dict, key=key)

    @classmethod
    def from_dataclass(
        cls,
        config: object,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Load configuration from a dataclass and instantiate the class.

        Automatically detects whether to use jsonargparse pattern (class_path/init_args)
        or direct instantiation based on the configuration structure.

        Args:
            config: Dataclass instance.
            key: Optional key to extract a sub-configuration from the dataclass.
            recursive: If True, recursively converts all nested dataclasses to dictionaries.
                If False, converts dataclass to dict non-recursively (preserves nested
                dataclass objects). Use recursive=False when your constructor expects
                dataclass objects (e.g., Feature, NormalizationParameters), and
                recursive=True when your constructor expects plain dictionaries.

        Returns:
            An instance of the class.

        Raises:
            TypeError: If config is not a dataclass instance.

        Examples:
            1. Jsonargparse pattern (auto-detected)
            ```python
            @dataclass
            class ModelConfig:
                class_path: str = "my.module.MyModel"
                init_args: dict = field(default_factory=lambda: {"hidden_size": 128, "num_layers": 3})

            model = MyModel.from_dataclass(ModelConfig())
            ```

            2. Direct instantiation (auto-detected)
            ```python
            @dataclass
            class ModelConfig:
                hidden_size: int = 128
                num_layers: int = 3

            model = MyModel.from_dataclass(ModelConfig())
            ```

            3. With recursive conversion control
            ```python
            # recursive=False: Preserves nested dataclass objects (default)
            # Use when constructor expects dataclass instances
            model = MyModel.from_dataclass(config, recursive=False)

            # recursive=True: Converts all nested dataclasses to dicts
            # Use when constructor expects plain dictionaries
            model = MyModel.from_dataclass(config, recursive=True)

            @dataclass
            class FeatureConfig:
                size: int = 128

            @dataclass
            class ModelConfig:
                feature: FeatureConfig = field(default_factory=FeatureConfig)

            # With recursive=False: feature remains as FeatureConfig instance
            # With recursive=True: feature becomes {"size": 128} dict
            ```
        """
        if not dataclasses.is_dataclass(config):
            msg = f"Expected dataclass instance, got {type(config)}"
            raise TypeError(msg)

        # Convert to dict and use from_dict logic
        if recursive:
            config_dict = dataclasses.asdict(config)  # type: ignore[arg-type]
        else:
            config_dict = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}

        return cls.from_dict(config_dict, key=key)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | BaseModel | object | str | Path,
        *,
        key: str | None = None,
        recursive: bool = False,
    ) -> Self:
        """Generic method to instantiate from any configuration format.

        This method automatically detects the configuration type and pattern,
        routing to the appropriate specific method.

        Args:
            config: Configuration in any supported format.
            key: Optional key to extract a sub-configuration.
            recursive: For dataclass and Pydantic configs, controls whether to convert
                nested objects to dicts (True) or preserve them (False, default).
                Ignored for dict and YAML configs.
                Ignored for other config types.

        Returns:
            An instance of the class.

        Raises:
            TypeError: If configuration type is unsupported.

        Examples:
            Auto-detects format and pattern
            ```python
            model = MyModel.from_config("config.yaml")
            model = MyModel.from_config(config_dict)
            model = MyModel.from_config(pydantic_config)
            model = MyModel.from_config(dataclass_config, recursive=True)
            ```
        """
        # Route to appropriate type-specific method
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

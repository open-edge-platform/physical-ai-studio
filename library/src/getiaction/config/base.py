# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base configuration class for policy configs.

Provides serialization methods:
- to_dict() / from_dict() - Plain dict for checkpoints (works with weights_only=True)
- to_jsonargparse() - Dict with class_path for dynamic instantiation
- save() / load() - YAML file I/O in jsonargparse format
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from getiaction.config.serializable import dataclass_to_dict, dict_to_dataclass

__all__ = ["Config"]


class Config:
    """Base class for policy configuration dataclasses.

    Provides serialization methods for policy configs:
    - `to_dict()` / `from_dict()` - Plain dict for checkpoints (works with weights_only=True)
    - `to_jsonargparse()` - Dict with class_path for dynamic instantiation
    - `save()` / `load()` - YAML file I/O in jsonargparse format

    All policy configs should:
    1. Be decorated with `@dataclass`
    2. Inherit from `Config`

    Example:
        ```python
        from dataclasses import dataclass
        from getiaction.config import Config

        @dataclass
        class MyPolicyConfig(Config):
            hidden_dim: int = 256
            num_layers: int = 4

        config = MyPolicyConfig(hidden_dim=512)

        # For checkpoints (plain dict)
        config_dict = config.to_dict()
        restored = MyPolicyConfig.from_dict(config_dict)

        # For YAML files
        config.save("config.yaml")
        restored = MyPolicyConfig.load("config.yaml")
        ```

    The serialization handles nested dataclasses, enums, optional types,
    dicts with dataclass values, and tuples.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert this config to a plain dict for serialization.

        Returns:
            Plain dictionary safe for torch.save with weights_only=True.

        Raises:
            TypeError: If this class is not a dataclass.
        """
        if not dataclasses.is_dataclass(self):
            msg = f"{self.__class__.__name__} must be a dataclass to use Config"
            raise TypeError(msg)

        result = dataclass_to_dict(self)
        if not isinstance(result, dict):
            msg = f"Expected dict from dataclass_to_dict, got {type(result)}"
            raise TypeError(msg)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Reconstruct this config from a dict.

        Uses type hints to reconstruct nested dataclasses without
        dynamic imports.

        Args:
            data: Dictionary representation of the config.

        Returns:
            Reconstructed config instance.

        Raises:
            TypeError: If this class is not a dataclass.
        """
        if not dataclasses.is_dataclass(cls):
            msg = f"{cls.__name__} must be a dataclass to use Config"
            raise TypeError(msg)

        return dict_to_dataclass(cls, data)

    def to_jsonargparse(self) -> dict[str, Any]:
        """Convert config to jsonargparse format for export metadata.

        This format includes the fully qualified class path, enabling
        dynamic instantiation via `instantiate_obj()`. Useful for
        export metadata where the config type needs to be recoverable
        without knowing it in advance.

        Returns:
            Dictionary with 'class_path' and 'init_args' keys.

        Example:
            ```python
            config = ACTConfig(chunk_size=50)
            jp_dict = config.to_jsonargparse()
            # {
            #     'class_path': 'getiaction.policies.act.config.ACTConfig',
            #     'init_args': {'chunk_size': 50, ...}
            # }

            # Can be instantiated dynamically:
            from getiaction.config import instantiate_obj
            restored = instantiate_obj(jp_dict)
            ```
        """
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "init_args": self.to_dict(),
        }

    def save(
        self,
        path: str | Path,
        *,
        format: Literal["jsonargparse", "dict"] = "jsonargparse",  # noqa: A002
    ) -> None:
        """Save config to a YAML file.

        Args:
            path: Path to save the config file (must have .yaml or .yml extension).
            format: Output format. "jsonargparse" (default) includes class_path and
                init_args for dynamic instantiation. "dict" saves the plain config
                without class metadata.

        Raises:
            ValueError: If the file extension is not .yaml or .yml.

        Example:
            ```python
            config = ACTConfig(chunk_size=50)

            # Save with class_path for dynamic instantiation
            config.save("config.yaml")
            config.save("config.yaml", format="jsonargparse")

            # Save plain dict (no class metadata)
            config.save("config.yaml", format="dict")
            ```
        """
        path = Path(path)
        data = self.to_dict() if format == "dict" else self.to_jsonargparse()

        if path.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {path.suffix}. Use .yaml or .yml"
            raise ValueError(msg)

        import yaml  # noqa: PLC0415

        with path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load config from a YAML file.

        Note: This method ignores the `class_path` in the file and uses the
        class it's called on. Use `instantiate_obj()` if you need dynamic
        class resolution.

        Args:
            path: Path to the config file (must have .yaml or .yml extension).

        Returns:
            Loaded config instance.

        Raises:
            ValueError: If the file extension is not .yaml or .yml.

        Example:
            ```python
            config = ACTConfig.load("config.yaml")
            ```
        """
        path = Path(path)

        if path.suffix not in {".yaml", ".yml"}:
            msg = f"Unsupported file extension: {path.suffix}. Use .yaml or .yml"
            raise ValueError(msg)

        import yaml  # noqa: PLC0415

        with path.open() as f:
            data = yaml.safe_load(f)

        # Handle both plain dict and jsonargparse format
        if "init_args" in data:
            data = data["init_args"]

        return cls.from_dict(data)

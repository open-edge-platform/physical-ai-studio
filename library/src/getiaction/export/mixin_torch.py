# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting and importing PyTorch models."""

import dataclasses
from copy import copy
from enum import StrEnum
from os import PathLike
from typing import Any, Self

import numpy as np
import torch
import yaml

from getiaction.config.instantiate import instantiate_obj_from_dict

GETIACTION_CONFIG_KEY = "getiaction_config"


class ToTorch:
    """Mixin class for exporting torch model checkpoints."""

    model: torch.nn.Module

    def to_torch(self, checkpoint_path: PathLike | str) -> None:
        """Export the model as a checkpoint with model configuration.

        This method saves the model's state dictionary along with its configuration
        to a checkpoint file. The configuration is embedded in the state dictionary
        under a special key for later retrieval.

        Args:
            checkpoint_path (PathLike | str): The file path where the checkpoint
                will be saved. Can be a string or path-like object.
            None: The method saves the checkpoint to disk and returns nothing.

        Note:
            - If the model has a 'config' attribute, it will be serialized and
              stored in the checkpoint.
            - The configuration is stored as YAML format under the
              GETIACTION_CONFIG_KEY in the state dictionary.
            - The saved checkpoint can be used to re-instantiate the model later.
        """
        state_dict = self.model.state_dict()
        config_dict = _serialize_model_config(self.model.config) if hasattr(self.model, "config") else {}
        state_dict[GETIACTION_CONFIG_KEY] = yaml.dump(config_dict, default_flow_style=False)

        torch.save(state_dict, checkpoint_path)  # nosec


class FromCheckpoint:
    """Mixin class for loading torch models from checkpoints."""

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict | PathLike | str,
    ) -> Self:
        """Load model state from a snapshot dictionary or file.

        This class method reconstructs a model instance from a snapshot containing
        the model's configuration and state dictionary. The snapshot can be provided
        either as a dictionary or as a path to a saved snapshot file.

        Args:
            snapshot (dict | PathLike | str): Either a dictionary containing the model's
                state_dict and configuration, or a path (string or PathLike object) to
                a saved snapshot file. When provided as a path, the snapshot is loaded
                using torch.load with CPU mapping and weights_only=True for security.

        Returns:
            Self: A new instance of the class initialized with the configuration from
                the snapshot.

        Raises:
            NotImplementedError: If the class does not implement the `from_dataclass`
                method, which is required for instantiation from the loaded configuration.

        Note:
            The snapshot must contain a configuration stored under the key defined by
            GETIACTION_CONFIG_KEY. This configuration is parsed as YAML and used to
            instantiate the appropriate dataclass configuration object, which is then
            passed to the `from_dataclass` method to create the model instance.
        """
        state_dict = {}
        if isinstance(snapshot, (str, PathLike)):
            state_dict = torch.load(snapshot, map_location="cpu", weights_only=True)
        else:
            state_dict = copy(snapshot)

        config = instantiate_obj_from_dict(yaml.safe_load(state_dict[GETIACTION_CONFIG_KEY]))
        state_dict.pop(GETIACTION_CONFIG_KEY)

        if hasattr(cls, "from_dataclass") and callable(cls.from_dataclass):  # type: ignore [attr-defined]
            return cls.from_dataclass(config)  # type: ignore [attr-defined]

        msg = "`FromCheckpoint` mixin requires the target class to implement `from_dataclass()` method."
        raise NotImplementedError(msg)


def _serialize_model_config(config: Any) -> dict:  # noqa: ANN401
    """Serialize a dataclass configuration object to a  yaml-friendly dictionary format.

    This function recursively converts a dataclass configuration object into a dictionary
    that can be serialized to JSON or other formats. It handles nested dataclasses,
    dictionaries containing dataclasses, StrEnum values, numpy arrays, and tuples.

    Args:
        config (Any): A dataclass configuration object to be serialized.

    Returns:
        dict: A dictionary containing two keys at the top level:
            - "class_path" (str): The fully qualified class name (module.classname).
            - "init_args" (dict): A dictionary of field names and their serialized values.

    Note:
        The function performs the following conversions:
        - Nested dataclasses are recursively serialized
        - Dataclasses within dictionaries are recursively serialized
        - StrEnum values are converted to strings
        - NumPy arrays are converted to lists using numpy's tolist() semantics
        - Tuples are converted to lists
    """
    config_dict: dict[str, Any] = {
        "class_path": f"{config.__class__.__module__}.{config.__class__.__qualname__}",
        "init_args": {field.name: getattr(config, field.name) for field in dataclasses.fields(config)},
    }

    updated_args: dict[Any, Any] = {}

    for k, v in config_dict["init_args"].items():
        target_k = str(k) if isinstance(k, StrEnum) else k

        if dataclasses.is_dataclass(v):
            updated_args[target_k] = _serialize_model_config(v)
        elif isinstance(v, dict):
            updated_inner_dict = {}
            for dk, dv in v.items():
                target_dk = str(dk) if isinstance(dk, StrEnum) else dk
                if dataclasses.is_dataclass(dv):
                    updated_inner_dict[target_dk] = _serialize_model_config(dv)
                else:
                    updated_inner_dict[target_dk] = dv
            updated_args[target_k] = updated_inner_dict
        elif isinstance(v, StrEnum):
            updated_args[target_k] = str(v)
        elif isinstance(v, np.ndarray):
            updated_args[target_k] = v.tolist()
        elif isinstance(v, tuple):
            updated_args[target_k] = list(v)
        else:
            updated_args[target_k] = v

    return {
        "class_path": config_dict["class_path"],
        "init_args": updated_args,
    }

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import StrEnum
from os import PathLike
from typing import Any, Self
from getiaction.config.instantiate import instantiate_obj_from_dict
import torch

import numpy as np
import yaml


class SnapshotIO:
    """Mixin class for saving/loading model snapshots."""

    GETIACTION_CONFIG_KEY = "getiaction_config"

    @property
    def config(self) -> Any:
        """Get the model configuration.

        Returns:
            A dataclass or dictionary containing the model's configuration.
        """
        return None

    def to_snapshot(self) -> dict:
        """Convert model state to a snapshot dictionary.

        Returns:
            A dictionary containing the model's state_dict.
        """

        state_dict = self.state_dict()
        config_dict = _serialize_model_config(self.config)
        if config_dict:
            state_dict[self.GETIACTION_CONFIG_KEY] = yaml.dump(config_dict, default_flow_style=False)
        else:
            state_dict[self.GETIACTION_CONFIG_KEY] = ""

        return state_dict

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict | PathLike | str
    ) -> Self:
        """Load model state from a snapshot dictionary.

        Args:
            snapshot: A dictionary containing the model's state_dict.
        """
        state_dict = {}
        if isinstance(snapshot, (str, PathLike)):
            state_dict = torch.load(snapshot, map_location="cpu", weights_only=True)
        else:
            state_dict = snapshot

        if SnapshotIO.GETIACTION_CONFIG_KEY in state_dict:
            config = instantiate_obj_from_dict(yaml.safe_load(state_dict[SnapshotIO.GETIACTION_CONFIG_KEY]))
            state_dict.pop(SnapshotIO.GETIACTION_CONFIG_KEY)

        return cls.from_dataclass(config)


def _serialize_model_config(config: Any) -> dict:
    """
    Serialize a dataclass configuration object to a yaml-friendly dictionary format.
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
        "init_args": {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    }

    updated_args: dict[Any, Any] = {}

    for k, v in config_dict["init_args"].items():
        if isinstance(k, StrEnum):
            k = str(k)

        if dataclasses.is_dataclass(v):
            updated_args[k] = _serialize_model_config(v)
        elif isinstance(v, dict):
            updated_inner_dict = {}
            for dk, dv in v.items():
                if isinstance(dk, StrEnum):
                    dk = str(dk)
                if dataclasses.is_dataclass(dv):
                    updated_inner_dict[dk] = _serialize_model_config(dv)
                else:
                    updated_inner_dict[dk] = dv
            updated_args[k] = updated_inner_dict
        elif isinstance(v, StrEnum):
            updated_args[k] = str(v)
        elif isinstance(v, np.ndarray):
            updated_args[k] = v.tolist()
        elif isinstance(v, tuple):
            updated_args[k] = list(v)
        else:
            updated_args[k] = v

    return {
        "class_path": config_dict["class_path"],
        "init_args": updated_args,
    }

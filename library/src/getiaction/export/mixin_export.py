# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting PyTorch models."""

import dataclasses
from enum import StrEnum
from os import PathLike
from typing import Any

import numpy as np
import torch
import yaml


CONFIG_KEY = "model_config"


class Export:
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
        state_dict[CONFIG_KEY] = yaml.dump(config_dict, default_flow_style=False)

        torch.save(state_dict, checkpoint_path)  # nosec

    def to_onnx(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None,
        **export_kwargs: Any,
    ) -> None:
        """Export the model to ONNX format.

        This method exports the model to the ONNX format using a provided input
        sample for tracing. Additional export options can be specified via keyword
        arguments.

        Args:
            onnx_path (PathLike | str): The file path where the ONNX model will be saved.
            input_sample (dict[str, torch.Tensor] | None): A sample input dictionary
        """
        if input_sample is None and hasattr(self.model, "get_sample_input") and callable(self.model.get_sample_input):
            input_sample = self.model.get_sample_input()
        else:
            msg = "An input sample must be provided for ONNX export, or the model must implement `get_sample_input()`."
            raise ValueError(msg)

        self.super().to_onnx(
            output_path,
            input_sample,
            **export_kwargs,
        )


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

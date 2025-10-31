# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting PyTorch models."""

import dataclasses
import inspect
from enum import StrEnum
from os import PathLike
from typing import Any

import numpy as np
import openvino
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

    @torch.no_grad()
    def to_onnx(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the model to ONNX format.

        This method exports the model to the ONNX format using a provided input
        sample for tracing. Additional export options can be specified via keyword
        arguments or through the model's `extra_export_args` property if it exists.

        Args:
            output_path (PathLike | str): The file path where the ONNX model will be saved.
            input_sample (dict[str, torch.Tensor] | None): A sample input dictionary.
                If `None`, the method will attempt to use the model's `sample_input`
                property. This input is used to trace the model during export.
            **export_kwargs: Additional keyword arguments to pass to `torch.onnx.export`.

        Raises:
            RuntimeError: If input sample is not provided and the model does not
                implement `sample_input` property.
        """
        if input_sample is None and hasattr(self.model, "sample_input"):
            input_sample = self.model.sample_input
        elif input_sample is None:
            msg = (
                "An input sample must be provided for ONNX export, or the model must implement `sample_input` property."
            )
            raise RuntimeError(msg)

        extra_model_args = self._get_export_extra_args("onnx")
        extra_model_args.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        self.model.eval()
        torch.onnx.export(
            self.model,
            args=(),
            kwargs={arg_name: input_sample},
            f=output_path,
            input_names=list(input_sample.keys()),
            **extra_model_args,
        )

    @torch.no_grad()
    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the model to OpenVINO format.

        Args:
            output_path (PathLike | str): Path where the OpenVINO model will be saved.
            input_sample (dict[str, torch.Tensor] | None, optional): Sample input tensor(s) for model tracing.
                If None, attempts to use the model's `sample_input` property. Defaults to None.
            **export_kwargs (dict): Additional keyword arguments to pass to the OpenVINO conversion process.

        Raises:
            RuntimeError: If no input sample is provided and the model does not implement a `sample_input` property.

        Notes:
            - The model is set to evaluation mode before conversion.
            - Output names can be specified in export_kwargs using the "output" key.
        """
        if input_sample is None and hasattr(self.model, "sample_input"):
            input_sample = self.model.sample_input
        elif input_sample is None:
            msg = "An input sample must be provided for OpenVINO export, or the model must implement "
            "`sample_input` property."
            raise RuntimeError(msg)

        extra_model_args = self._get_export_extra_args("openvino")
        extra_model_args.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        output_names = extra_model_args.get("output", None)
        if output_names is not None:
            extra_model_args.pop("output")

        input_shapes = [openvino.Shape(tuple(tensor.shape)) for tensor in input_sample.values()]

        self.model.eval()

        ov_model = openvino.convert_model(
            self.model,
            example_input={arg_name: input_sample},
            input=input_shapes,
            **extra_model_args,
        )
        _postprocess_openvino_model(ov_model, output_names)

        openvino.save_model(ov_model, output_path)

    @torch.no_grad()
    def to_torch_export_ir(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the model to Torch Export IR format.

        This method exports the model to Torch Export IR (Intermediate Representation) format,
        which can be used for deployment and for further optimization and inference via executorch or similar tools.

        Args:
            output_path (PathLike | str): The file path where the exported Torch IR model will be saved.
            input_sample (dict[str, torch.Tensor] | None, optional): A sample input tensor dictionary
                to trace the model. If None, the method will attempt to use the model's
                `sample_input` property. Defaults to None.
            **export_kwargs (dict): Additional keyword arguments to pass to the torch.export.export function.

        Raises:
            RuntimeError: If no input sample is provided and the model does not have a `sample_input` property.

        Note:
            - The model is set to evaluation mode before export.
            - The export uses strict mode by default.
            - Additional export arguments can be specified through the model's export configuration
              and will be merged with the provided export_kwargs.
        """
        if input_sample is None and hasattr(self.model, "sample_input"):
            input_sample = self.model.sample_input
        elif input_sample is None:
            msg = (
                "An input sample must be provided for ONNX export, or the model must implement `sample_input` property."
            )
            raise RuntimeError(msg)

        extra_model_args = self._get_export_extra_args("torch_ir")
        extra_model_args.update(export_kwargs)

        self.model.eval()
        torch_program = torch.export.export(
            self.model,
            args=(input_sample,),
            **extra_model_args,
        )

        torch.export.save(torch_program, output_path)  # nosec

    def _get_export_extra_args(self, backend: str) -> dict[str, Any]:
        """Retrieve extra export arguments for a specific format.

        This method checks if the model has an `extra_export_args` property and
        retrieves any additional export arguments for the specified format.

        Args:
            backend (str): The export backend (e.g., "onnx", "openvino").

        Returns:
            dict[str, Any]: A dictionary of extra export arguments for the specified backend.
                Returns an empty dictionary if no extra arguments are found.
        """
        extra_model_args: dict[str, Any] = {}
        if hasattr(self.model, "extra_export_args") and backend in self.model.extra_export_args:
            extra_model_args = self.model.extra_export_args[backend]
        return extra_model_args

    def _get_forward_arg_name(self) -> str:
        """Get the name of the first positional argument of the model's forward method.

        This method inspects the signature of the model's forward method and returns
        the name of the first positional argument (excluding 'self').

        Returns:
            str: The name of the first positional argument in the forward method.
        """
        sig = inspect.signature(self.model.forward)
        positional_args = [
            param_name
            for param_name, param in sig.parameters.items()
            if param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY}
            and param_name != "self"
        ]

        return next(iter(positional_args))


def _postprocess_openvino_model(ov_model: openvino.Model, output_names: list[str] | None) -> None:
    """Postprocess an OpenVINO model by setting output tensor names.

    This function handles two scenarios:
    1. Workaround for OpenVINO Converter (OVC) bug where a single output model
        doesn't have a name assigned to its output tensor.
    2. Assigns custom output names to the model's output tensors when provided.
    The naming process follows a similar approach to PyTorch's ONNX export.

    Args:
            ov_model (openvino.Model): The OpenVINO model to postprocess.
            output_names (list[str] | None): Optional list of custom names to assign
                to the model's output tensors. If provided and the model has at least
                as many outputs as names in the list, the names will be assigned to
                the corresponding output tensors in order.


    Note:
            - If a single output exists without a name, it will be named "output1".
            - When output_names is provided, only the first len(output_names) outputs
            will be renamed, even if the model has more outputs.
    """
    if len(ov_model.outputs) == 1 and len(ov_model.outputs[0].get_names()) == 0:
        # workaround for OVC's bug: single output doesn't have a name in OV model
        ov_model.outputs[0].tensor.set_names({"output1"})

    # name assignment process is similar to torch onnx export
    if output_names is not None and len(ov_model.outputs) >= len(output_names):
        for i, name in enumerate(output_names):
            ov_model.outputs[i].tensor.set_names({name})


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

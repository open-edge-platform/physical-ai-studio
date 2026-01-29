# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting PyTorch models."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import lightning
    import openvino
    import torch
except ImportError as e:
    msg = "Export dependencies not installed.\nInstall with: pip install getiaction[train]"
    raise ImportError(msg) from e

import yaml

from getiaction import __version__
from getiaction.export.types import ExportBackend

if TYPE_CHECKING:
    from os import PathLike

CONFIG_KEY = "model_config"
POLICY_NAME_KEY = "policy_name"
DATASET_STATS_KEY = "dataset_stats"


class Export:
    """Mixin class for exporting torch model checkpoints."""

    model: torch.nn.Module

    def _create_metadata(
        self,
        export_dir: Path,
        backend: ExportBackend,
        **metadata_kwargs: dict,
    ) -> None:
        """Create metadata files for exported model.

        Args:
            export_dir: Directory containing exported model
            backend: Export backend used
            **metadata_kwargs: Additional metadata to include

        Raises:
            TypeError: If ``metadata_extra`` is present but not a mapping.
            ValueError: If ``metadata_extra`` contains keys that collide with existing metadata.
        """
        # Build metadata
        metadata = {
            "getiaction_version": __version__,
            "policy_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "backend": str(backend),
            **metadata_kwargs,
        }

        # Add model config if available
        if hasattr(self.model, "config") and hasattr(self.model.config, "to_jsonargparse"):
            metadata["config"] = self.model.config.to_jsonargparse()

        metadata_extra = getattr(self, "metadata_extra", None)
        if metadata_extra is not None:
            if not isinstance(metadata_extra, Mapping):
                msg = f"metadata_extra must be a mapping, got: {type(metadata_extra)!r}"
                raise TypeError(msg)

            collisions = set(metadata_extra) & set(metadata)
            if collisions:
                msg = f"metadata_extra collides with existing metadata keys: {sorted(collisions)}"
                raise ValueError(msg)

            metadata.update(metadata_extra)

        # Save as YAML (preferred)
        yaml_path = export_dir / "metadata.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(metadata, f, default_flow_style=False)

    def _prepare_export_path(self, output_path: PathLike | str, extension: str) -> Path:
        """Prepare export path, handling both directory and file paths.

        Args:
            output_path: Directory or file path for export
            extension: File extension to use (e.g., ".xml", ".onnx", ".pt")

        Returns:
            Path: Complete file path with proper extension
        """
        path = Path(output_path)

        # For torch checkpoints, accept both .pt and .pth extensions
        valid_extensions = [extension]
        if extension == ".pt":
            valid_extensions.append(".pth")

        # If path is a directory or doesn't have a valid extension, add filename
        if path.is_dir() or (not path.suffix or path.suffix not in valid_extensions):
            # Use policy name for filename
            policy_name = self.__class__.__name__.lower()
            path /= f"{policy_name}{extension}"

        # Create parent directory
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def to_torch(self, checkpoint_path: PathLike | str) -> None:
        """Export the model as a checkpoint with model configuration.

        This method saves the model's state dictionary along with its configuration
        to a checkpoint file. The configuration is embedded in the state dictionary
        under a special key for later retrieval.

        Args:
            checkpoint_path: Path where the checkpoint will be saved.

        Note:
            - If the model has a 'config' attribute, it will be serialized and
              stored in the checkpoint.
            - The configuration is stored as YAML format under the
              GETIACTION_CONFIG_KEY in the state dictionary.
            - The saved checkpoint can be used to re-instantiate the model later.
        """
        model_path = self._prepare_export_path(checkpoint_path, ".pt")
        export_dir = model_path.parent

        checkpoint = {}
        checkpoint["state_dict"] = self.state_dict() if hasattr(self, "state_dict") else {}

        if hasattr(self.model, "config"):
            config_dict = self.model.config.to_dict()
            checkpoint[CONFIG_KEY] = config_dict
        elif hasattr(self, "hparams"):
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            checkpoint["pytorch-lightning_version"] = lightning.__version__
            checkpoint["loops"] = {}
            checkpoint["hparams_name"] = "kwargs"
            checkpoint["hyper_parameters"] = dict(self.hparams)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, str(model_path))  # nosec B614

        # Create metadata files
        self._create_metadata(export_dir, ExportBackend.TORCH)

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
            output_path (PathLike | str): Directory or file path where the ONNX model will be saved.
                If directory, creates {policy_name}.onnx. If file, uses as-is.
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

        model_path = self._prepare_export_path(output_path, ".onnx")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.ONNX)
        extra_model_args.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        self.model.eval()
        torch.onnx.export(
            self.model,
            args=(),
            kwargs={arg_name: input_sample},
            f=str(model_path),
            input_names=list(input_sample.keys()),
            **extra_model_args,
        )

        # Create metadata files
        self._create_metadata(export_dir, ExportBackend.ONNX)

    @torch.no_grad()
    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the model to OpenVINO format.

        Args:
            output_path (PathLike | str): Directory or file path where the OpenVINO model will be saved.
                If directory, creates {policy_name}.xml. If file, uses as-is.
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

        model_path = self._prepare_export_path(output_path, ".xml")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.OPENVINO)
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

        openvino.save_model(ov_model, str(model_path))

        # Create metadata files
        self._create_metadata(export_dir, ExportBackend.OPENVINO)

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
            output_path (PathLike | str): Directory or file path where the exported Torch IR model will be saved.
                If directory, creates {policy_name}.pt2. If file, uses as-is.
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
                "An input sample must be provided for Torch Export IR export, "
                "or the model must implement `sample_input` property."
            )
            raise RuntimeError(msg)

        model_path = self._prepare_export_path(output_path, ".pt2")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.TORCH_EXPORT_IR)
        extra_model_args.update(export_kwargs)

        self.model.eval()
        torch_program = torch.export.export(
            self.model,
            args=(input_sample,),
            **extra_model_args,
        )

        torch.export.save(torch_program, str(model_path))  # nosec

        # Create metadata files
        self._create_metadata(export_dir, ExportBackend.TORCH_EXPORT_IR)

    def export(
        self,
        output_path: PathLike | str,
        backend: ExportBackend | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: dict,
    ) -> None:
        """Export the model to the specified backend format.

        This method serves as a unified interface for exporting the model to different
        formats by dispatching to the appropriate backend-specific export method.

        Args:
            output_path (PathLike | str): The file path where the exported model will be saved.
            backend (ExportBackend | str): The export backend to use.
                Can be an ExportBackend enum value or a string
                ("onnx", "openvino", "torch_export_ir").
            input_sample (dict[str, torch.Tensor] | None, optional): A sample
                input tensor dictionary for model tracing.
                If None, attempts to use the model's `sample_input` property.
                Defaults to None.
            **export_kwargs (dict): Additional keyword arguments to pass to the
                backend-specific export method.

        Raises:
            ValueError: If an unsupported backend is specified.
        """
        backend = ExportBackend(backend)

        if backend == ExportBackend.ONNX:
            self.to_onnx(output_path, input_sample, **export_kwargs)
        elif backend == ExportBackend.OPENVINO:
            self.to_openvino(output_path, input_sample, **export_kwargs)
        elif backend == ExportBackend.TORCH_EXPORT_IR:
            self.to_torch_export_ir(output_path, input_sample, **export_kwargs)
        elif backend == ExportBackend.TORCH:
            self.to_torch(output_path)
        else:
            msg = f"Unsupported export backend: {backend}"
            raise ValueError(msg)

    def _get_export_extra_args(self, backend: ExportBackend | str) -> dict[str, Any]:
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

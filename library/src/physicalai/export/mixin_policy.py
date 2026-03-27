# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting Policies."""

import inspect
import tempfile
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import lightning
import onnx
import openvino
import openvino_tokenizers
import torch
import yaml
from onnxruntime_extensions import gen_processing_models

from physicalai.export.backends import (
    ExecuTorchExportParameters,
    ExportBackend,
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
)
from physicalai.inference.manifest import (
    ComponentSpec,
    Manifest,
    PolicySpec,
)
from physicalai.inference.runners.action_chunking import ActionChunking
from physicalai.inference.runners.single_pass import SinglePass
from physicalai.train import __version__

from .mixin_model import ExportableModelMixin

CONFIG_KEY = "model_config"
POLICY_NAME_KEY = "policy_name"
DATASET_STATS_KEY = "dataset_stats"


class ExportablePolicyMixin:
    """Mixin class for exporting torch model checkpoints."""

    model: ExportableModelMixin
    _preprocessor: torch.nn.Module

    def _create_metadata(
        self,
        export_dir: Path,
        backend: ExportBackend,
        **metadata_kwargs: dict,
    ) -> None:
        """Create metadata files for exported model.

        Writes both ``manifest.json`` (new structured format) and
        ``metadata.yaml`` (legacy format) for backward compatibility.

        Args:
            export_dir: Directory containing exported model
            backend: Export backend used
            **metadata_kwargs: Additional metadata to include

        Raises:
            TypeError: If ``metadata_extra`` is present but not a mapping.
            ValueError: If ``metadata_extra`` contains keys that collide with existing metadata.
        """
        policy_class = f"{self.__class__.__module__}.{self.__class__.__name__}"

        metadata = {
            "physicalai_train_version": __version__,
            "policy_class": policy_class,
            "backend": str(backend),
            **metadata_kwargs,
        }

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

        yaml_path = export_dir / "metadata.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False)

        manifest = self._build_manifest(metadata, backend)
        manifest.save(export_dir / "manifest.json")

    def _build_manifest(self, metadata: dict[str, Any], backend: ExportBackend) -> Manifest:
        """Build a ``Manifest`` from the collected metadata.

        Args:
            metadata: Flat metadata dict (already includes metadata_extra).
            backend: Export backend used.

        Returns:
            Structured manifest ready for serialisation.
        """
        policy_class = metadata.get("policy_class", "")
        policy_name = self.__class__.__name__.lower()

        use_action_queue = metadata.get("use_action_queue", False)
        chunk_size = metadata.get("chunk_size", 1)
        kind = "action_chunking" if use_action_queue else "single_pass"

        if use_action_queue:
            runner = ComponentSpec.from_class(
                ActionChunking,
                runner=ComponentSpec.from_class(SinglePass),
                chunk_size=chunk_size,
            )
        else:
            runner = ComponentSpec.from_class(SinglePass)

        artifact_filename = f"{policy_name}{backend.extension}"

        return Manifest(
            policy=PolicySpec(
                name=policy_name,
                kind=kind,
                class_path=policy_class,
            ),
            artifacts={str(backend): artifact_filename},
            runner=runner,
        )

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

        Raises:
            NotImplementedError: If Torch export is not supported by the policy.
        """
        if ExportBackend.TORCH not in self.supported_export_backends:
            msg = (
                f"Torch export is not implemented for this policy. Supported backends: {self.supported_export_backends}"
            )
            raise NotImplementedError(msg)

        model_path = self._prepare_export_path(checkpoint_path, ".pt")
        export_dir = model_path.parent

        checkpoint = {}
        checkpoint["state_dict"] = self.state_dict() if hasattr(self, "state_dict") else {}

        if hasattr(self, "hparams"):
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
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.
            NotImplementedError: If ONNX export is not supported by the model.
        """
        if ExportBackend.ONNX not in self.supported_export_backends:
            msg = (
                f"ONNX export is not implemented for this policy. Supported backends: {self.supported_export_backends}"
            )
            raise NotImplementedError(msg)

        if input_sample is None:
            input_sample = self._get_default_export_input_sample()

        if input_sample is None:
            msg = "An input sample must be provided for ONNX export, or the model must implement "
            "`sample_input` property."
            raise RuntimeError(msg)

        model_path = self._prepare_export_path(output_path, ".onnx")
        export_dir = model_path.parent

        extra_model_args = cast("ONNXExportParameters", self._get_export_extra_args(ExportBackend.ONNX))
        extra_export_kwargs = extra_model_args.exporter_kwargs
        extra_export_kwargs.update(export_kwargs)

        arg_name = self._get_forward_arg_name()

        self.model.eval()
        self._onnx_core_export_step(
            model_path=model_path,
            input_sample=input_sample,
            arg_name=arg_name,
            **extra_export_kwargs,
        )

        if extra_model_args.export_tokenizer:
            onnx_tokenizer = gen_processing_models(
                self._preprocessor.exportable_tokenizer,
                pre_kwargs={
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self._preprocessor.max_token_len,
                },
            )[0]
            if onnx_tokenizer is not None:
                onnx.save(onnx_tokenizer, export_dir / "tokenizer.onnx")
            else:
                msg = (
                    "Failed to convert tokenizer to ONNX format. The tokenizer may not be compatible with ONNX export."
                )
                raise RuntimeError(msg)

        # Create metadata files
        self._create_metadata(export_dir, ExportBackend.ONNX, preprocessing_type=extra_model_args.preprocessing_type)

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
            RuntimeError: If input sample is not provided and the model does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.

        Notes:
            - The model is set to evaluation mode before conversion.
            - Output names can be specified in export_kwargs using the "output" key.

        Raises:
            RuntimeError: If input sample is not provided and the model does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.
            NotImplementedError: If OpenVINO export is not supported by the policy.
        """
        if ExportBackend.OPENVINO not in self.supported_export_backends:
            msg = (
                f"OpenVINO export is not implemented for this policy.\n"
                f"Supported backends: {self.supported_export_backends}"
            )
            raise NotImplementedError(msg)

        if input_sample is None:
            input_sample = self._get_default_export_input_sample()

        if input_sample is None:
            msg = "An input sample must be provided for OpenVINO export, or the model must implement "
            "`sample_input` property."
            raise RuntimeError(msg)

        model_path = self._prepare_export_path(output_path, ".xml")
        export_dir = model_path.parent

        arg_name = self._get_forward_arg_name()
        input_shapes = [openvino.Shape(tuple(tensor.shape)) for tensor in input_sample.values()]

        extra_model_args: OpenVINOExportParameters = cast(
            "OpenVINOExportParameters",
            self._get_export_extra_args(ExportBackend.OPENVINO),
        )
        extra_export_kwargs = extra_model_args.exporter_kwargs

        if extra_model_args.via_onnx:
            onnx_model_args = cast("ONNXExportParameters", self._get_export_extra_args(ExportBackend.ONNX))
            extra_export_kwargs = onnx_model_args.exporter_kwargs

        extra_export_kwargs.update(export_kwargs)

        self.model.eval()

        if extra_model_args.via_onnx:
            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
                self._onnx_core_export_step(
                    model_path=Path(tmp.name),
                    input_sample=input_sample,
                    arg_name=arg_name,
                    **extra_export_kwargs,
                )
                ov_model = openvino.convert_model(tmp.name)
        else:
            ov_model = openvino.convert_model(
                self.model,
                example_input={arg_name: input_sample},
                input=input_shapes,
                **extra_export_kwargs,
            )
        _postprocess_openvino_model(ov_model, extra_model_args.outputs)

        openvino.save_model(ov_model, str(model_path), compress_to_fp16=extra_model_args.compress_to_fp16)

        if extra_model_args.export_tokenizer:
            ov_tokenizer = openvino_tokenizers.convert_tokenizer(
                self._preprocessor.exportable_tokenizer,
                with_detokenizer=False,
                max_length=self._preprocessor.max_token_len,
                use_max_padding=True,
            )
            if ov_tokenizer is not None:
                openvino.save_model(ov_tokenizer, export_dir / "tokenizer.xml")
            else:
                msg = (
                    "Failed to convert tokenizer to OpenVINO format. "
                    "The tokenizer may not be compatible with OpenVINO export."
                )
                raise RuntimeError(msg)

        self._create_metadata(
            export_dir,
            ExportBackend.OPENVINO,
            preprocessing_type=extra_model_args.preprocessing_type,
        )

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

        Raises:
            RuntimeError: If input sample is not provided and the model does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.
            NotImplementedError: If Torch Export IR is not supported by the policy.
        """
        if ExportBackend.TORCH_EXPORT_IR not in self.supported_export_backends:
            msg = (
                f"Torch Export IR is not implemented for this policy. "
                f"Supported backends: {self.supported_export_backends}"
            )
            raise NotImplementedError(msg)

        if input_sample is None:
            input_sample = self._get_default_export_input_sample()

        if input_sample is None:
            msg = (
                "An input sample must be provided for Torch Export IR export, "
                "or the model must implement `sample_input` property."
            )
            raise RuntimeError(msg)

        model_path = self._prepare_export_path(output_path, ".pt2")
        export_dir = model_path.parent

        extra_model_args = self._get_export_extra_args(ExportBackend.TORCH_EXPORT_IR).exporter_kwargs
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

    @torch.no_grad()
    def to_executorch(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        *,
        delegate: str | None = None,
        delegate_config: dict[str, Any] | None = None,
        **export_kwargs: dict,
    ) -> Path:
        """Export the model to ExecuTorch format.

        Args:
            output_path: Directory or file path where the ExecuTorch model will be saved.
                If directory, creates ``{policy_name}.pte``. If file, uses as-is.
            input_sample: A sample input tensor dictionary used to trace/export the model.
                If ``None``, attempts to use the model's ``sample_input`` property.
            delegate: ExecuTorch delegate backend to use. Defaults to ``None``
                (portable mode, no delegation). Supported values:

                - ``None``: Portable mode — no delegation, uses ExecuTorch portable ops.
                - ``"xnnpack"``: XNNPACK delegation — optimized CPU kernels for ARM/x86.
                  Works out-of-the-box with ``pip install executorch``.
                - ``"openvino"``: OpenVINO delegation — requires ``nncf`` for export and a
                  custom-built ExecuTorch runtime with OpenVINO backend for inference.
            delegate_config: Optional delegate-specific configuration. For ``"openvino"``,
                supports ``{"device": "CPU"}`` (or other supported target device).
            **export_kwargs: Additional keyword arguments passed to ``torch.export.export``.

        Returns:
            Path: Path to the exported ``.pte`` model file.

        Raises:
            RuntimeError: If input sample is not provided and the model does not
                implement ``sample_input`` property.
            ImportError: If the required ``executorch`` package (or selected delegate
                dependencies) is not installed.
            ValueError: If an unsupported delegate is specified.
        """
        if input_sample is None and hasattr(self.model, "sample_input"):
            input_sample = self.model.sample_input
        elif input_sample is None:
            msg = (
                "An input sample must be provided for ExecuTorch export, "
                "or the model must implement `sample_input` property."
            )
            raise RuntimeError(msg)

        model_path = self._prepare_export_path(output_path, ".pte")
        export_dir = model_path.parent

        extra_model_args = cast(
            "ExecuTorchExportParameters",
            self._get_export_extra_args(ExportBackend.EXECUTORCH),
        )
        extra_export_kwargs = extra_model_args.exporter_kwargs
        extra_export_kwargs.update(export_kwargs)

        if delegate is None:
            delegate = extra_model_args.delegate

        try:
            from executorch.exir import to_edge_transform_and_lower  # noqa: PLC0415
        except ImportError as e:
            msg = "executorch package is required for ExecuTorch export. Install with: pip install executorch"
            raise ImportError(msg) from e

        self.model.eval()
        aten_dialect = torch.export.export(
            self.model,
            args=(input_sample,),
            **extra_export_kwargs,
        )

        try:
            if delegate == "openvino":
                from executorch.backends.openvino.partitioner import OpenvinoPartitioner  # noqa: PLC0415
                from executorch.exir.backend.backend_details import CompileSpec  # noqa: PLC0415

                compile_spec = [CompileSpec("device", (delegate_config or {}).get("device", "CPU").encode())]
                partitioner = OpenvinoPartitioner(compile_spec)
            elif delegate == "xnnpack":
                from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: PLC0415
                    XnnpackPartitioner,
                )

                partitioner = XnnpackPartitioner()
            elif delegate is None or delegate == "portable":
                partitioner = None
            else:
                msg = (
                    f"Unsupported ExecuTorch delegate: {delegate!r}. "
                    f"Supported delegates: 'portable', 'openvino', 'xnnpack', None"
                )
                raise ValueError(msg)
        except ImportError as e:
            msg = f"ExecuTorch delegate dependencies are required for delegate={delegate!r}."
            raise ImportError(msg) from e

        if partitioner is not None:
            edge_program = to_edge_transform_and_lower(aten_dialect, partitioner=[partitioner])
        else:
            edge_program = to_edge_transform_and_lower(aten_dialect)

        exec_program = edge_program.to_executorch()

        with model_path.open("wb") as f:
            exec_program.write_to_file(f)

        # Create metadata files
        self._create_metadata(
            export_dir,
            ExportBackend.EXECUTORCH,
            input_names=list(input_sample.keys()),  # type: ignore[arg-type, union-attr]
        )

        return model_path

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
        elif backend == ExportBackend.EXECUTORCH:
            self.to_executorch(output_path, input_sample, **export_kwargs)
        elif backend == ExportBackend.TORCH:
            self.to_torch(output_path)
        else:
            msg = f"Unsupported export backend: {backend}"
            raise ValueError(msg)

    def _onnx_core_export_step(
        self,
        model_path: Path,
        input_sample: dict[str, torch.Tensor],
        arg_name: str,
        **export_kwargs: dict,
    ) -> None:
        """Run torch.onnx.export and save the model to a file.

        Args:
            model_path: Path where the ONNX model will be saved.
            input_sample: Input tensors for tracing.
            arg_name: Name of the forward method's first positional argument.
            **export_kwargs: Additional keyword arguments for torch.onnx.export.
        """
        torch.onnx.export(
            self.model,
            args=(),
            kwargs={arg_name: input_sample},
            f=str(model_path),
            input_names=list(input_sample.keys()),
            **export_kwargs,
        )

    def _get_default_export_input_sample(self) -> dict[str, torch.Tensor] | None:
        """Retrieve a default export input sample for the model.

        This method attempts to obtain a sample input from the model if available,
        processes it through the preprocessor, and filters the result to return only
        torch.Tensor values.

        Returns:
            dict[str, torch.Tensor] | None: A dictionary containing string keys mapped to
                torch.Tensor values extracted from the processed sample input. Returns None
                if the model does not have a 'sample_input' attribute.
        """
        processed_sample = self._preprocessor(self.model.sample_input)
        return {k: v for k, v in processed_sample.items() if isinstance(v, torch.Tensor)}

    def _get_export_extra_args(self, backend: ExportBackend | str) -> ExportParameters:
        """Retrieve extra export arguments for a specific format.

        This method checks if the model has an `extra_export_args` property and
        retrieves any additional export arguments for the specified format.

        Args:
            backend (str): The export backend (e.g., "onnx", "openvino").

        Returns:
            ExportParameters: Extra export arguments for the specified backend.
                Returns an empty ExportParameters instance if no extra arguments are found.
        """
        if backend in self.model.extra_export_args:
            return self.model.extra_export_args[backend]
        return ExportBackend(backend).parameter_class()

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

    @property
    def supported_export_backends(self) -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH]


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

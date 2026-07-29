# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin classes for exporting Policies."""

import inspect
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any, cast

import lightning
import openvino
import openvino_tokenizers
import torch
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype
from physicalai.inference.manifest import (
    ComponentSpec,
    Manifest,
    ModelSpec,
    PolicySource,
    PolicySpec,
)
from physicalai.inference.runners.single_pass import SinglePass

from physicalai.export.backends import (
    ExecuTorchDelegate,
    ExecuTorchExportParameters,
    ExportBackend,
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)

CONFIG_KEY = "model_config"
POLICY_NAME_KEY = "policy_name"
DATASET_STATS_KEY = "dataset_stats"


class ExportablePolicyMixin:
    """Mixin class for exporting torch model checkpoints.

    Attributes:
        model: Internal torch module to be exported.
        _preprocessor: The torch module applied to raw inputs before they reach
            the model during export tracing.
        device: The torch device on which the model and sample inputs live.
    """

    model: torch.nn.Module
    _preprocessor: torch.nn.Module
    device: torch.device

    @property
    def sample_input(self) -> dict[str, torch.Tensor | str] | None:
        """Return a sample input dictionary used to trace the model during export.

        Override in subclasses that support non-Torch export backends to provide
        example tensors matching the model's expected input format. Returning
        ``None`` (the default) means the policy does not provide a sample input,
        and callers must supply one explicitly when invoking export methods that
        require tracing.

        Returns:
            A dictionary mapping input names to example tensors (or strings, for
            non-tensor inputs such as task descriptions), or ``None``.
        """
        if self.inputs_schema is None:
            return None

        input_sample: dict[str, torch.Tensor | str] = {}
        for feature in self.inputs_schema:
            if feature.dtype is InferenceFeatureDtype.STRING:
                input_sample[feature.name] = "Example prompt string"
                continue
            # Scalar features (empty shape) are emitted as 0-d tensors, without
            # a leading batch dimension; tensor-shaped features get a batch of 1.
            shape = feature.shape if feature.shape == () else (1, *feature.shape)
            if feature.dtype is InferenceFeatureDtype.INT64:
                input_sample[feature.name] = torch.zeros(shape, dtype=torch.int64)
            else:
                input_sample[feature.name] = torch.randn(shape, dtype=torch.float32)

        return input_sample

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Provide a description of model's expected inputs.

        Override in subclasses to return a list of InferenceFeature objects describing
        the model's expected inputs. This information can be used by export methods to
        generate appropriate input samples for tracing when a sample input is not explicitly provided.
        The default implementation returns None, indicating that no input schema is provided.

        Returns:
            A list of InferenceFeature describing the model's expected inputs, or None if not provided.
        """
        return None

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Provide a description of model's expected outputs.

        Override in subclasses to return a list of InferenceFeature objects describing
        the model's expected outputs. This information can be used by export methods to
        generate appropriate output metadata in the manifest.
        The default implementation returns None, indicating that no output schema is provided.

        Returns:
            A list of InferenceFeature describing the model's expected outputs, or None if not provided.
        """
        return None

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Return extra arguments for the export process.

        Override in subclasses to provide backend-specific export parameters.

        Returns:
            A dictionary mapping backend names to their export parameters.
        """
        return {}

    @contextmanager
    def _scoped_rtc(self, *, enable: bool) -> Generator[None, None, None]:
        """Temporarily set enable_rtc on the model, restoring the previous value on exit."""
        prev = getattr(self.model, "enable_rtc", False)
        setattr(self.model, "enable_rtc", enable)  # noqa: B010
        try:
            yield
        finally:
            setattr(self.model, "enable_rtc", prev)  # noqa: B010

    def create_manifest(
        self,
        export_dir: Path,
        backend: ExportBackend,
        runner: ComponentSpec,
        preprocessors: list[ComponentSpec] | None = None,
        postprocessors: list[ComponentSpec] | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        **extras: Any,  # noqa: ANN401
    ) -> None:
        """Create ``manifest.json`` for an exported model.

        Args:
            export_dir: Directory containing the exported model.
            backend: Export backend used.
            runner: Runner component spec to include in the manifest.
            preprocessors: Preprocessor component specs to include in the manifest.
            postprocessors: Postprocessor component specs to include in the manifest.
            input_names: Optional ordered list of model input names.
            output_names: Optional ordered list of model output names.
            **extras: Additional keyword arguments to forward to the manifest.
        """
        policy_class = f"{self.__class__.__module__}.{self.__class__.__name__}"
        policy_name = self.__class__.__name__.lower()
        artifact_filename = f"{policy_name}{backend.extension}"

        if input_names is not None:
            extras["input_names"] = list(input_names)
        if output_names is not None:
            extras["output_names"] = list(output_names)

        manifest = Manifest(
            policy=PolicySpec(
                name=policy_name,
                source=PolicySource(class_path=policy_class),
            ),
            model=ModelSpec(
                runner=runner,
                artifacts={str(backend): artifact_filename},
                preprocessors=preprocessors or [],
                postprocessors=postprocessors or [],
                input_features=extras.pop("input_features", []),
                output_features=extras.pop("output_features", []),
            ),
            **extras,
        )
        manifest.save(export_dir / "manifest.json")

    @staticmethod
    def _to_component_specs(items: list[Any]) -> list[ComponentSpec]:
        """Convert items to ``ComponentSpec`` instances.

        ``ComponentSpec`` instances pass through unchanged. Dataclass instances
        (such as ``InferenceFeature``) are converted by introspecting their
        fields and capturing the fully-qualified class path.

        Args:
            items: List of ``ComponentSpec`` or dataclass instances.

        Returns:
            List of ``ComponentSpec`` instances ready for manifest serialization.
        """
        specs: list[ComponentSpec] = []
        for item in items:
            if isinstance(item, ComponentSpec):
                specs.append(item)
                continue
            cls = type(item)
            class_path = f"{cls.__module__}.{cls.__qualname__}"
            init_args = {field: getattr(item, field) for field in getattr(item, "__dataclass_fields__", {})}
            specs.append(ComponentSpec(class_path=class_path, init_args=init_args))
        return specs

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
        if ExportBackend.TORCH not in self.get_supported_export_backends():
            msg = (
                "Torch export is not implemented for this policy. "
                f"Supported backends: {self.get_supported_export_backends()}"
            )
            raise NotImplementedError(msg)

        model_path = self._prepare_export_path(checkpoint_path, ".pt")
        export_dir = model_path.parent

        extra_model_args: TorchExportParameters = cast(
            "TorchExportParameters",
            self._get_export_extra_args(ExportBackend.TORCH),
        )

        checkpoint = {}
        checkpoint["state_dict"] = self.state_dict() if hasattr(self, "state_dict") else {}

        if hasattr(self, "hparams"):
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            checkpoint["pytorch-lightning_version"] = lightning.__version__
            checkpoint["loops"] = {}
            checkpoint["hparams_name"] = "kwargs"
            checkpoint["hyper_parameters"] = dict(self.hparams)

        if hasattr(self, "on_save_checkpoint"):
            self.on_save_checkpoint(checkpoint)

        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        torch.save(checkpoint, str(model_path))  # nosec B614

        self.create_manifest(
            export_dir,
            ExportBackend.TORCH,
            runner=ComponentSpec.from_class(SinglePass),
            preprocessors=extra_model_args.preprocessors_specs,
            postprocessors=extra_model_args.postprocessors_specs,
            input_features=self._to_component_specs(self.inputs_schema or []),
            output_features=self._to_component_specs(self.outputs_schema or []),
        )

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
                If `None`, the method will attempt to use the policy's `sample_input`
                property. This input is used to trace the model during export.
            **export_kwargs: Additional keyword arguments to pass to `torch.onnx.export`.

        Raises:
            RuntimeError: If input sample is not provided and the policy does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.
            NotImplementedError: If ONNX export is not supported by the model or ONNX tokenizer export is requested.
        """
        if ExportBackend.ONNX not in self.get_supported_export_backends():
            msg = (
                "ONNX export is not implemented for this policy. "
                f"Supported backends: {self.get_supported_export_backends()}"
            )
            raise NotImplementedError(msg)

        enable_rtc = bool(export_kwargs.pop("enable_rtc", False))
        with self._scoped_rtc(enable=enable_rtc):
            if input_sample is None:
                input_sample = self._get_default_export_input_sample()

            if input_sample is None:
                msg = "An input sample must be provided for ONNX export, or the policy must implement "
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
                msg = "Tokenizer export is not supported for ONNX backend at this time."
                raise NotImplementedError(msg)

            self.create_manifest(
                export_dir,
                ExportBackend.ONNX,
                runner=ComponentSpec.from_class(SinglePass),
                preprocessors=extra_model_args.preprocessors_specs,
                postprocessors=extra_model_args.postprocessors_specs,
                input_features=self._to_component_specs(self.inputs_schema or []),
                output_features=self._to_component_specs(self.outputs_schema or []),
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
            output_path (PathLike | str): Directory or file path where the OpenVINO model will be saved.
                If directory, creates {policy_name}.xml. If file, uses as-is.
            input_sample (dict[str, torch.Tensor] | None, optional): Sample input tensor(s) for model tracing.
                If None, attempts to use the policy's `sample_input` property. Defaults to None.
            **export_kwargs (dict): Additional keyword arguments to pass to the OpenVINO conversion process.

        Raises:
            RuntimeError: If input sample is not provided and the policy does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.

        Notes:
            - The model is set to evaluation mode before conversion.
            - Output names can be specified in export_kwargs using the "output" key.

        Raises:
            RuntimeError: If input sample is not provided and the policy does not
                implement `sample_input` property. Also if export is failed due to other issues
                like wrong export options.
            NotImplementedError: If OpenVINO export is not supported by the policy.
        """
        if ExportBackend.OPENVINO not in self.get_supported_export_backends():
            msg = (
                f"OpenVINO export is not implemented for this policy.\n"
                f"Supported backends: {self.get_supported_export_backends()}"
            )
            raise NotImplementedError(msg)

        enable_rtc = bool(export_kwargs.pop("enable_rtc", False))
        with self._scoped_rtc(enable=enable_rtc):
            if input_sample is None:
                input_sample = self._get_default_export_input_sample()

            if input_sample is None:
                msg = "An input sample must be provided for OpenVINO export, or the policy must implement "
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
                    ov_model = openvino.convert_model(
                        tmp.name,
                        example_input={arg_name: input_sample},
                        input=input_shapes,
                    )
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
                self._preprocessor.tokenizer,
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

        self.create_manifest(
            export_dir,
            ExportBackend.OPENVINO,
            runner=ComponentSpec.from_class(SinglePass),
            preprocessors=extra_model_args.preprocessors_specs,
            postprocessors=extra_model_args.postprocessors_specs,
            input_features=self._to_component_specs(self.inputs_schema or []),
            output_features=self._to_component_specs(self.outputs_schema or []),
        )

    @torch.no_grad()
    def to_executorch(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        *,
        delegate: ExecuTorchDelegate | None = None,
        delegate_config: dict[str, Any] | None = None,
        **export_kwargs: dict,
    ) -> Path:
        """Export the model to ExecuTorch format.

        Args:
            output_path: Directory or file path where the ExecuTorch model will be saved.
                If directory, creates ``{policy_name}.pte``. If file, uses as-is.
            input_sample: A sample input tensor dictionary used to trace/export the model.
                If ``None``, attempts to use the policy's ``sample_input`` property.
            delegate: ExecuTorch delegate backend to use. Defaults to ``None``
                (uses value from ``ExecuTorchExportParameters``). Supported values:

                - ``"portable"``: Portable mode — no delegation, uses ExecuTorch portable ops.
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
            NotImplementedError: If ExecuTorch export is not supported by the policy.
            RuntimeError: If input sample is not provided and the policy does not
                implement ``sample_input`` property.
            ImportError: If the required ``executorch`` package (or selected delegate
                dependencies) is not installed.
            ValueError: If an unsupported delegate is specified.
        """
        if ExportBackend.EXECUTORCH not in self.get_supported_export_backends():
            msg = (
                f"ExecuTorch export is not implemented for this policy.\n"
                f"Supported backends: {self.get_supported_export_backends()}"
            )
            raise NotImplementedError(msg)

        if input_sample is None:
            input_sample = self._get_default_export_input_sample()
        if input_sample is None:
            msg = (
                "An input sample must be provided for ExecuTorch export, "
                "or the policy must implement the `sample_input` property."
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

        self.create_manifest(
            export_dir,
            ExportBackend.EXECUTORCH,
            runner=ComponentSpec.from_class(SinglePass),
            input_names=list(input_sample.keys()),  # type: ignore[arg-type, union-attr]
            output_names=extra_model_args.output_names,
            input_features=self._to_component_specs(self.inputs_schema or []),
            output_features=self._to_component_specs(self.outputs_schema or []),
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
                ("onnx", "openvino", "executorch", "torch").
            input_sample (dict[str, torch.Tensor] | None, optional): A sample
                input tensor dictionary for model tracing.
                If None, attempts to use the policy's `sample_input` property.
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

        This method attempts to obtain a sample input from the policy if available,
        processes it through the preprocessor, and filters the result to return only
        torch.Tensor values.

        Returns:
            dict[str, torch.Tensor] | None: A dictionary containing string keys mapped to
                torch.Tensor values extracted from the processed sample input. Returns None
                if the policy does not provide a `sample_input`.
        """
        sample = self.sample_input
        if sample is None:
            return None
        sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        processed_sample = self._preprocessor(sample)
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
        if backend in self.extra_export_args:
            return self.extra_export_args[backend]
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

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
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

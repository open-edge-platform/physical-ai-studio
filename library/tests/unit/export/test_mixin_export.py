# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mixin_export module."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import onnx
import pytest
import torch

from physicalai.export.backends import ExportParameters, ONNXExportParameters, OpenVINOExportParameters
from physicalai.export.mixin_policy import ExportablePolicyMixin, ExportBackend


# Test configurations
@dataclass
class SimpleConfig:
    """Simple configuration for testing."""

    input_dim: int = 10
    output_dim: int = 5


# Test models
class SimpleModel(torch.nn.Module):
    """Simple PyTorch model for testing."""

    def __init__(self, config: SimpleConfig):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(config.input_dim, config.output_dim)

    def forward(self, x):
        return self.linear(x)


class ModelWithSampleInput(torch.nn.Module):
    """Model implementing sample_input property."""

    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        # batch is a dict passed as the first parameter
        return self.linear(batch["input_tensor"])

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate sample input."""
        return {"input_tensor": torch.randn(1, self.input_dim)}


class ModelWithExtraExportArgs(torch.nn.Module):
    """Model implementing extra_export_args property."""

    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        # batch is a dict passed as the first parameter
        return self.linear(batch["x"])

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate sample input."""
        return {"x": torch.randn(1, self.input_dim)}

    @property
    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Extra ONNX export arguments."""
        return {
            "onnx": ONNXExportParameters(
                exporter_kwargs={"output_names": ["output"]},
            ),
        }


class ModelWithMultipleInputs(torch.nn.Module):
    """Model with multiple inputs in the dict."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(5, 10)
        self.combine = torch.nn.Linear(20, 8)

    def forward(self, batch):
        # batch is a dict containing multiple tensors
        x1 = self.linear1(batch["input_a"])
        x2 = self.linear2(batch["input_b"])
        combined = torch.cat([x1, x2], dim=-1)
        return self.combine(combined)

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate sample input."""
        return {
            "input_a": torch.randn(1, 5),
            "input_b": torch.randn(1, 5),
        }


class ModelWithDictInput(torch.nn.Module):
    """Model accepting dict input (single parameter)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, batch):
        # batch is expected to be a dict
        return self.linear(batch["data"])

    @property
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Generate sample input."""
        return {"data": torch.randn(1, 10)}


class IdentityPreprocessor(torch.nn.Module):
    """Identity preprocessor that returns input as-is."""

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return x


class ExportWrapper(ExportablePolicyMixin):
    """Wrapper class for testing Export mixin."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._preprocessor = IdentityPreprocessor()
        if not hasattr(model, "extra_export_args"):
            model.extra_export_args = {
                ExportBackend.ONNX: ONNXExportParameters(),
                ExportBackend.OPENVINO: OpenVINOExportParameters(),
            }

    def _get_default_export_input_sample(self) -> dict[str, torch.Tensor] | None:
        if not hasattr(self.model, "sample_input"):
            return None
        return super()._get_default_export_input_sample()

    @property
    def metadata_extra(self) -> dict[str, Any]:
        return {"chunk_size": 10, "use_action_queue": True}

    @classmethod
    def get_supported_export_backends(cls) -> list[str | ExportBackend]:
        return [ExportBackend.ONNX, ExportBackend.OPENVINO, ExportBackend.EXECUTORCH]


class TestToOnnx:
    """Tests for to_onnx method."""

    def test_to_onnx_with_sample_input_from_model(self, tmp_path):
        """Test ONNX export using model's sample_input property."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.to_onnx(output_path)

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model can be loaded
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_to_onnx_with_provided_input_sample(self, tmp_path):
        """Test ONNX export with explicitly provided input sample."""
        model = SimpleModel(SimpleConfig(input_dim=8, output_dim=4))

        # Wrap the model with a forward that accepts batch dict
        class WrappedModel(torch.nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner = inner_model

            def forward(self, batch):
                return self.inner(batch["x"])

        wrapped = WrappedModel(model)
        wrapper = ExportWrapper(wrapped)

        input_sample = {"x": torch.randn(1, 8)}
        output_path = tmp_path / "model.onnx"

        wrapper.to_onnx(output_path, input_sample=input_sample)

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_to_onnx_kwargs_override_model_args(self, tmp_path):
        """Test that provided kwargs override model's extra_export_args."""
        model = ModelWithExtraExportArgs(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        # Override the output_names from the model
        wrapper.to_onnx(output_path, output_names=["custom_output"])

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Check that custom output name is used
        output_names = [output.name for output in onnx_model.graph.output]
        assert "custom_output" in output_names

    def test_to_onnx_with_multiple_inputs(self, tmp_path):
        """Test ONNX export with model having multiple inputs."""
        model = ModelWithMultipleInputs()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.to_onnx(output_path)

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Check that both inputs are in the model
        input_names = [input.name for input in onnx_model.graph.input]
        assert "input_a" in input_names
        assert "input_b" in input_names

    def test_to_onnx_with_dict_input(self, tmp_path):
        """Test ONNX export with model accepting dict as single parameter."""
        model = ModelWithDictInput()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.to_onnx(output_path)

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_to_onnx_without_sample_input_raises_error(self, tmp_path):
        """Test that RuntimeError is raised when no input sample is provided."""
        # Model without sample_input property
        model = SimpleModel(SimpleConfig())
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        with pytest.raises(RuntimeError, match="input sample must be provided"):
            wrapper.to_onnx(output_path)

    def test_to_onnx_via_export_method(self, tmp_path):
        """Test ONNX export using the generic export method."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.export(backend="onnx", output_path=output_path)

        assert output_path.exists()
        assert ExportBackend.ONNX in wrapper.get_supported_export_backends()

        # Verify the ONNX model can be loaded
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)


class TestToOpenVINO:
    """Tests for to_openvino method."""

    def test_to_openvino_with_sample_input_from_model(self, tmp_path):
        """Test OpenVINO export using model's sample_input property."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert ExportBackend.OPENVINO in wrapper.get_supported_export_backends()
        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_default_export_args(self, tmp_path):
        """Test that provided kwargs override model's extra_export_args."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)

        model.extra_export_args = {}
        wrapper = ExportWrapper(model)
        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert output_path.exists()
        assert ExportBackend.OPENVINO in wrapper.get_supported_export_backends()

    def test_to_openvino_with_provided_input_sample(self, tmp_path):
        """Test OpenVINO export with explicitly provided input sample."""
        model = SimpleModel(SimpleConfig(input_dim=8, output_dim=4))

        # Wrap the model with a forward that accepts batch dict
        class WrappedModel(torch.nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner = inner_model

            def forward(self, batch):
                return self.inner(batch["x"])

        wrapped = WrappedModel(model)
        wrapper = ExportWrapper(wrapped)

        input_sample = {"x": torch.randn(1, 8)}
        output_path = tmp_path / "model.xml"

        wrapper.to_openvino(output_path, input_sample=input_sample)

        assert ExportBackend.OPENVINO in wrapper.get_supported_export_backends()
        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_with_multiple_inputs(self, tmp_path):
        """Test OpenVINO export with model having multiple inputs."""
        model = ModelWithMultipleInputs()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert ExportBackend.OPENVINO in wrapper.get_supported_export_backends()
        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_with_dict_input(self, tmp_path):
        """Test OpenVINO export with model accepting dict as single parameter."""
        model = ModelWithDictInput()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert ExportBackend.OPENVINO in wrapper.get_supported_export_backends()
        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_without_sample_input_raises_error(self, tmp_path):
        """Test that RuntimeError is raised when no input sample is provided."""
        # Model without sample_input property
        model = SimpleModel(SimpleConfig())
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"

        with pytest.raises(RuntimeError, match="input sample must be provided"):
            wrapper.to_openvino(output_path)

    @pytest.mark.parametrize("fp16", [True, False])
    def test_to_openvino_via_export_method(self, tmp_path, fp16):
        """Test OpenVINO export using the generic export method."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        model.extra_export_args = {
            "openvino": OpenVINOExportParameters(
                compress_to_fp16=fp16,
            ),
        }
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.export(backend="openvino", output_path=output_path)

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_via_onnx(self, tmp_path):
        """Test OpenVINO export via ONNX intermediate model."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        model.extra_export_args = {
            ExportBackend.OPENVINO: OpenVINOExportParameters(
                via_onnx=True,
            ),
        }
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()


class TestToExecutorch:
    """Tests for to_executorch method."""

    def _mock_executorch_modules(self):
        """Create mock modules for executorch lazy imports.

        Returns a dict of mock modules and key mock objects for assertions.
        """
        mock_exir = MagicMock()
        mock_to_edge = MagicMock()
        mock_exir.to_edge_transform_and_lower = mock_to_edge

        mock_edge_program = MagicMock()
        mock_to_edge.return_value = mock_edge_program

        mock_exec_program = MagicMock()
        mock_edge_program.to_executorch.return_value = mock_exec_program

        mock_openvino_partitioner_mod = MagicMock()
        mock_backend_details_mod = MagicMock()

        modules = {
            "executorch": MagicMock(),
            "executorch.exir": mock_exir,
            "executorch.backends": MagicMock(),
            "executorch.backends.openvino": MagicMock(),
            "executorch.backends.openvino.partitioner": mock_openvino_partitioner_mod,
            "executorch.exir.backend": MagicMock(),
            "executorch.exir.backend.backend_details": mock_backend_details_mod,
        }

        return {
            "modules": modules,
            "mock_to_edge": mock_to_edge,
            "mock_edge_program": mock_edge_program,
            "mock_exec_program": mock_exec_program,
            "mock_openvino_partitioner_mod": mock_openvino_partitioner_mod,
            "mock_backend_details_mod": mock_backend_details_mod,
        }

    def test_to_executorch_happy_path(self, tmp_path):
        """Test full ExecuTorch export flow with mocked executorch modules."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        mocks = self._mock_executorch_modules()

        with patch.dict("sys.modules", mocks["modules"]), patch("torch.export.export") as mock_torch_export:
            mock_torch_export.return_value = MagicMock()  # aten_dialect

            result = wrapper.to_executorch(tmp_path / "model.pte")

            # Assert write_to_file was called (writes .pte content)
            mocks["mock_exec_program"].write_to_file.assert_called_once()

            # Assert metadata.yaml was created
            assert (tmp_path / "metadata.yaml").exists()

            # Assert .pte file was created (open() creates it even with mocked write)
            assert (tmp_path / "model.pte").exists()

            # Verify metadata contains input_names
            import yaml

            with open(tmp_path / "metadata.yaml") as f:
                metadata = yaml.safe_load(f)
            assert "input_names" in metadata

            assert result == tmp_path / "model.pte"

    def test_to_executorch_no_sample_input(self, tmp_path):
        """Test that RuntimeError is raised when model has no sample_input."""
        model = SimpleModel(SimpleConfig())
        wrapper = ExportWrapper(model)

        with pytest.raises(RuntimeError, match="input sample"):
            wrapper.to_executorch(tmp_path / "model.pte")

    def test_to_executorch_import_error(self, tmp_path):
        """Test that ImportError is raised when executorch is not installed."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        with patch.dict("sys.modules", {"executorch.exir": None}), pytest.raises(ImportError):
            wrapper.to_executorch(tmp_path / "model.pte")

    def test_to_executorch_unsupported_delegate(self, tmp_path):
        """Test that ValueError is raised for unsupported delegate."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        mocks = self._mock_executorch_modules()

        with (
            patch.dict("sys.modules", mocks["modules"]),
            patch("torch.export.export", return_value=MagicMock()),
            pytest.raises(ValueError, match="Unsupported"),
        ):
            wrapper.to_executorch(tmp_path / "model.pte", delegate="unsupported_delegate")

    def test_to_executorch_no_delegate(self, tmp_path):
        """Test ExecuTorch export in portable mode (no partitioner)."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        mocks = self._mock_executorch_modules()

        with patch.dict("sys.modules", mocks["modules"]), patch("torch.export.export") as mock_torch_export:
            mock_torch_export.return_value = MagicMock()

            wrapper.to_executorch(tmp_path / "model.pte", delegate=None)

            # Assert to_edge_transform_and_lower was called without partitioner kwarg
            mocks["mock_to_edge"].assert_called_once()
            call_args = mocks["mock_to_edge"].call_args
            assert "partitioner" not in (call_args.kwargs or {})

    def test_to_executorch_custom_delegate_config(self, tmp_path):
        """Test ExecuTorch export with custom delegate configuration."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        mocks = self._mock_executorch_modules()
        mock_compile_spec = mocks["mock_backend_details_mod"].CompileSpec

        with patch.dict("sys.modules", mocks["modules"]), patch("torch.export.export") as mock_torch_export:
            mock_torch_export.return_value = MagicMock()

            wrapper.to_executorch(tmp_path / "model.pte", delegate="openvino", delegate_config={"device": "GPU"})

            # Assert CompileSpec was called with ("device", b"GPU")
            mock_compile_spec.assert_called_once_with("device", b"GPU")

    def test_export_dispatches_to_executorch(self, tmp_path):
        """Test that export() dispatcher calls to_executorch()."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        with patch.object(wrapper, "to_executorch") as mock_to_executorch:
            wrapper.export(backend=ExportBackend.EXECUTORCH, output_path=tmp_path / "model.pte")
            mock_to_executorch.assert_called_once()

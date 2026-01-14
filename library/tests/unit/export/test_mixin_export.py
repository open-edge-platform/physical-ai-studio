# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for mixin_export module."""

from dataclasses import dataclass
from typing import Any

import onnx
import pytest
import torch

from getiaction.export.mixin_export import Export, ExportBackend


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
    def extra_export_args(self) -> dict:
        """Extra ONNX export arguments."""
        return {
            "onnx": {
                "output_names": ["output"],
                "dynamic_axes": {"x": {0: "batch_size"}, "output": {0: "batch_size"}},
            }
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


class ExportWrapper(Export):
    """Wrapper class for testing Export mixin."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @property
    def metadata_extra(self) -> dict[str, Any]:
        return {"chunk_size": 10, "use_action_queue": True}


class TestToOnnx:
    """Tests for to_onnx method."""

    def test_to_onnx_with_sample_input_from_model(self, tmp_path):
        """Test ONNX export using model's sample_input property."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.to_onnx(output_path)

        assert output_path.exists()

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

        # Verify the ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_to_onnx_without_sample_input_raises_error(self, tmp_path):
        """Test that RuntimeError is raised when no input sample is provided."""
        # Model without sample_input property
        model = SimpleModel(SimpleConfig())
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"

        with pytest.raises(RuntimeError, match="input sample must be provided"):
            wrapper.to_onnx(output_path)

    def test_to_onnx_via_export_method(self, tmp_path):
        """Test ONNX export using the generic export method."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.onnx"
        wrapper.export(backend="onnx", output_path=output_path)

        assert output_path.exists()

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

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

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

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_with_multiple_inputs(self, tmp_path):
        """Test OpenVINO export with model having multiple inputs."""
        model = ModelWithMultipleInputs()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()

    def test_to_openvino_with_dict_input(self, tmp_path):
        """Test OpenVINO export with model accepting dict as single parameter."""
        model = ModelWithDictInput()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.to_openvino(output_path)

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

    def test_to_openvino_via_export_method(self, tmp_path):
        """Test OpenVINO export using the generic export method."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.xml"
        wrapper.export(backend="openvino", output_path=output_path)

        assert output_path.exists()
        assert (tmp_path / "model.bin").exists()


class TestToTorchExportIR:
    """Tests for to_torch_export_ir method."""

    def test_to_torch_export_ir_with_sample_input_from_model(self, tmp_path):
        """Test TorchIR export using model's sample_input property."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"
        wrapper.to_torch_export_ir(output_path)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path) # nosec
        assert loaded_program is not None

    def test_to_torch_export_ir_with_provided_input_sample(self, tmp_path):
        """Test TorchIR export with explicitly provided input sample."""
        model = ModelWithDictInput()
        wrapper = ExportWrapper(model)

        input_sample = {"data": torch.randn(1, 10)}
        output_path = tmp_path / "model.pt2"

        wrapper.to_torch_export_ir(output_path, input_sample=input_sample)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path) # nosec
        assert loaded_program is not None

    def test_to_torch_export_ir_kwargs_override_model_args(self, tmp_path):
        """Test that provided kwargs override model's extra_export_args."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)

        # Add extra_export_args for torch_ir
        model.extra_export_args = {
            "torch_ir": {
                "strict": True,
            }
        }

        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"
        # Override the strict mode from the model
        wrapper.to_torch_export_ir(output_path, strict=False)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path) # nosec
        assert loaded_program is not None

    def test_to_torch_export_ir_with_multiple_inputs(self, tmp_path):
        """Test TorchIR export with model having multiple inputs."""
        model = ModelWithMultipleInputs()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"
        wrapper.to_torch_export_ir(output_path)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path) # nosec
        assert loaded_program is not None

    def test_to_torch_export_ir_with_dict_input(self, tmp_path):
        """Test TorchIR export with model accepting dict as single parameter."""
        model = ModelWithDictInput()
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"
        wrapper.to_torch_export_ir(output_path)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path) # nosec
        assert loaded_program is not None

    def test_to_torch_export_ir_without_sample_input_raises_error(self, tmp_path):
        """Test that RuntimeError is raised when no input sample is provided."""
        # Model without sample_input property
        model = SimpleModel(SimpleConfig())
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"

        with pytest.raises(RuntimeError, match="input sample must be provided"):
            wrapper.to_torch_export_ir(output_path)

    def test_to_torch_export_ir_model_in_eval_mode(self, tmp_path):
        """Test that model is set to eval mode during TorchIR export."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        # Set model to training mode
        model.train()
        assert model.training is True

        output_path = tmp_path / "model.pt2"
        wrapper.to_torch_export_ir(output_path)

        # Model should be in eval mode after export
        assert model.training is False

    def test_to_torch_export_ir_via_export_method(self, tmp_path):
        """Test TorchIR export using the generic export method."""
        model = ModelWithSampleInput(input_dim=10, output_dim=5)
        wrapper = ExportWrapper(model)

        output_path = tmp_path / "model.pt2"
        wrapper.export(backend=ExportBackend.TORCH_EXPORT_IR, output_path=output_path)

        assert output_path.exists()

        # Verify the exported program can be loaded
        loaded_program = torch.export.load(output_path)  # nosec
        assert loaded_program is not None

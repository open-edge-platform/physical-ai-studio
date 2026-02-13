# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference adapters."""

from pathlib import Path
import inspect
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from getiaction.export.mixin_export import ExportBackend
from getiaction.inference.adapters import (
    ONNXAdapter,
    OpenVINOAdapter,
    RuntimeAdapter,
    TorchExportAdapter,
    TorchAdapter,
    get_adapter,
)


class TestGetAdapter:
    """Test adapter factory function."""

    @pytest.mark.parametrize("backend_type", [ExportBackend, str])
    @pytest.mark.parametrize(
        ("backend_name", "expected_type"),
        [
            ("openvino", OpenVINOAdapter),
            ("onnx", ONNXAdapter),
            ("torch_export_ir", TorchExportAdapter),
            ("torch", TorchAdapter),
        ],
    )
    def test_get_adapter(
        self,
        backend_type: type,
        backend_name: str,
        expected_type: type[RuntimeAdapter],
    ) -> None:
        """Test get_adapter returns correct type for both enum and string inputs."""
        backend = backend_type(backend_name) if backend_type == ExportBackend else backend_name
        adapter = get_adapter(backend)
        assert isinstance(adapter, expected_type)

    def test_invalid_backend(self) -> None:
        """Test invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid ExportBackend"):
            get_adapter("invalid_backend")


class TestOpenVINOAdapter:
    """Test OpenVINO inference adapter."""

    def test_lifecycle(self, tmp_path: Path) -> None:
        """Test complete adapter lifecycle: init, load, predict."""
        # Setup
        model_path = tmp_path / "model.xml"
        model_path.touch()
        (tmp_path / "model.bin").touch()

        mock_model = MagicMock()
        mock_input, mock_output = Mock(), Mock()
        mock_input.any_name, mock_output.any_name = "input", "output"
        mock_model.inputs, mock_model.outputs = [mock_input], [mock_output]
        mock_model.return_value = [np.array([[1.0, 2.0]])]

        mock_ov = MagicMock()
        mock_ov.Core.return_value.compile_model.return_value = mock_model

        # Test init, load, and predict
        with patch.dict("sys.modules", {"openvino": mock_ov}):
            adapter = OpenVINOAdapter(device="CPU")
            assert adapter.device == "CPU"
            assert "CPU" in repr(adapter)

            adapter.load(model_path)
            assert adapter.input_names == ["input"]
            assert adapter.output_names == ["output"]

            outputs = adapter.predict({"input": np.array([[1.0, 2.0]])})
            assert "output" in outputs and isinstance(outputs["output"], np.ndarray)

    def test_error_cases(self, tmp_path: Path) -> None:
        """Test error handling for file not found, missing dependency, and predict without load."""
        adapter = OpenVINOAdapter()

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            adapter.load(Path("/nonexistent/model.xml"))

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict({"input": np.array([1.0])})

        model_path = tmp_path / "model.xml"
        model_path.touch()
        with patch.dict("sys.modules", {"openvino": None}):
            with pytest.raises(ImportError, match="OpenVINO is not installed"):
                adapter.load(model_path)


class TestONNXAdapter:
    """Test ONNX Runtime inference adapter."""

    @pytest.mark.parametrize(
        ("device", "expected_providers"),
        [
            ("cpu", ["CPUExecutionProvider"]),
            ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
            ("tensorrt", ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]),
        ],
    )
    def test_provider_selection(self, device: str, expected_providers: list[str]) -> None:
        """Test execution provider selection by device."""
        assert ONNXAdapter(device=device)._get_providers() == expected_providers

    def test_lifecycle(self, tmp_path: Path) -> None:
        """Test complete adapter lifecycle."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        mock_session = MagicMock()
        mock_input, mock_output = Mock(), Mock()
        mock_input.name, mock_output.name = "input", "output"
        mock_session.get_inputs.return_value, mock_session.get_outputs.return_value = [mock_input], [mock_output]
        mock_session.run.return_value = [np.array([[1.0, 2.0]])]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            adapter = ONNXAdapter(device="cpu")
            adapter.load(model_path)
            assert adapter.input_names == ["input"] and adapter.output_names == ["output"]

            outputs = adapter.predict({"input": np.array([[1.0]])})
            assert "output" in outputs

    def test_error_cases(self, tmp_path: Path) -> None:
        """Test error handling."""
        adapter = ONNXAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load(Path("/nonexistent/model.onnx"))

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict({"input": np.array([1.0])})

        model_path = tmp_path / "model.onnx"
        model_path.touch()
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="ONNX Runtime is not installed"):
                adapter.load(model_path)


class TestTorchAdapter:
    """Test Torch inference adapter."""

    def test_lifecycle(self, tmp_path: Path) -> None:
        """Test complete adapter lifecycle: init, load, predict."""

        model_path = tmp_path / "model.pt"
        metadata_path = tmp_path / "metadata.yaml"
        model_path.touch()
        metadata_path.touch()

        with metadata_path.open("w") as f:
            f.write("policy_class: getiaction.policies.act.ACT\n")

        mock_model = MagicMock()
        mock_model.model.return_value = torch.tensor([[1.0, 2.0]])
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.model.sample_input = {"input": torch.tensor([[0.0]])}
        mock_model.model.extra_export_args = {"torch": {"output_names": ["output"], "input_names": ["observation"]}}

        with patch("getiaction.policies.act.ACT.load_from_checkpoint", return_value=mock_model):
            adapter = TorchAdapter(device="cpu")
            assert adapter.torch_device == torch.device("cpu")
            assert "cpu" in repr(adapter)

            adapter.load(model_path)

            with patch(
                "getiaction.inference.adapters.torch.TorchAdapter._convert_outputs_to_numpy",
                return_value={"output": np.array([[1.0, 2.0]])},
            ):
                # Test with numpy arrays (standard adapter interface)
                outputs = adapter.predict({
                    "images": np.random.randn(1, 3, 224, 224),
                    "state": np.random.randn(1, 2),
                })
                assert "output" in outputs and isinstance(outputs["output"], np.ndarray)

    def test_error_cases(self, tmp_path: Path) -> None:
        """Test error handling for file not found and predict without load."""
        adapter = TorchAdapter()

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            adapter.load(Path("/nonexistent/model.pt"))

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict({"state": np.array([1.0])})

    def test_predict_with_raw_observation_dict(self, tmp_path: Path) -> None:
        """Test predict accepts raw unpacked observation dicts."""
        model_path = tmp_path / "model.pt"
        metadata_path = tmp_path / "metadata.yaml"
        model_path.touch()
        metadata_path.touch()

        with metadata_path.open("w") as f:
            f.write("policy_class: getiaction.policies.act.ACT\n")

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.model.extra_export_args = {"torch": {"output_names": ["output"], "input_names": ["observation"]}}

        with patch("getiaction.policies.act.ACT.load_from_checkpoint", return_value=mock_model):
            adapter = TorchAdapter(device="cpu")
            adapter.load(model_path)

            with patch(
                "getiaction.inference.adapters.torch.TorchAdapter._convert_outputs_to_numpy",
                return_value={"output": np.array([[1.0, 2.0]])},
            ):
                outputs = adapter.predict({"state": np.array([[1.0]])})
                assert "output" in outputs

    def test_load_failure(self, tmp_path: Path) -> None:
        """Test error handling when torch.load fails."""
        model_path = tmp_path / "model.pt"
        metadata_path = tmp_path / "metadata.yaml"
        model_path.touch()
        metadata_path.touch()

        with metadata_path.open("w") as f:
            f.write("policy_class: getiaction.policies.act.ACT\n")

        with patch("getiaction.policies.act.ACT.load_from_checkpoint", side_effect=RuntimeError("Load error")):
            adapter = TorchAdapter()
            with pytest.raises(RuntimeError, match="Failed to load"):
                adapter.load(model_path)

    def test_device_selection(self) -> None:
        """Test device selection for Torch adapter."""
        adapter_cpu = TorchAdapter(device="cpu")
        assert adapter_cpu.torch_device == torch.device("cpu")

        adapter_cuda = TorchAdapter(device="cuda")
        assert adapter_cuda.torch_device == torch.device("cuda")

    def test_input_output_names_before_load(self) -> None:
        """Test input/output names return empty lists before model is loaded."""
        adapter = TorchAdapter()
        adapter._policy = MagicMock()

        assert adapter.input_names == []
        assert adapter.output_names == []


class TestTorchExportAdapter:
    """Test Torch Export IR adapter."""

    def test_lifecycle(self, tmp_path: Path) -> None:
        """Test complete adapter lifecycle."""
        import torch

        model_path = tmp_path / "model.pt2"
        model_path.touch()

        mock_program = MagicMock()
        mock_module = MagicMock()
        mock_module.return_value = {"output": torch.tensor([[1.0, 2.0]])}
        mock_program.module.return_value = mock_module

        # Mock call_spec for input names
        mock_dict_spec = Mock()
        mock_dict_spec.context = ["input"]
        mock_in_spec = Mock()
        mock_in_spec.children_specs = [Mock(children_specs=[mock_dict_spec])]
        mock_program.call_spec.in_spec = mock_in_spec

        # Mock graph_signature for output names
        mock_sig = Mock()
        mock_sig.user_outputs = ["output"]
        mock_program.graph_signature = mock_sig

        with patch("torch.export.load", return_value=mock_program):
            adapter = TorchExportAdapter()
            adapter.load(model_path)

            assert adapter.input_names == ["input"] and adapter.output_names == ["output"]

            outputs = adapter.predict({"input": np.array([[1.0]])})
            assert "output" in outputs and isinstance(outputs["output"], np.ndarray)

    def test_error_cases(self, tmp_path: Path) -> None:
        """Test error handling."""
        adapter = TorchExportAdapter()

        with pytest.raises(FileNotFoundError):
            adapter.load(Path("/nonexistent/model.pt2"))

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict({"input": np.array([1.0])})

        model_path = tmp_path / "model.pt2"
        model_path.touch()

        with patch("torch.export.load", side_effect=RuntimeError("Load error")):
            with pytest.raises(RuntimeError, match="Failed to load"):
                adapter.load(model_path)

        # Test missing inputs
        mock_program = MagicMock()
        mock_program.module.return_value = MagicMock()

        # Mock call_spec for input names
        mock_dict_spec = Mock()
        mock_dict_spec.context = ["input1", "input2"]
        mock_in_spec = Mock()
        mock_in_spec.children_specs = [Mock(children_specs=[mock_dict_spec])]
        mock_program.call_spec.in_spec = mock_in_spec

        # Mock graph_signature for output names
        mock_program.graph_signature.user_outputs = ["output"]

        with patch("torch.export.load", return_value=mock_program):
            adapter.load(model_path)
            with pytest.raises(ValueError, match="Missing required inputs"):
                adapter.predict({"input1": np.array([1.0])})


class TestRuntimeAdapter:
    """Test RuntimeAdapter base class."""

    @pytest.fixture
    def concrete_adapter_class(self):
        """Fixture providing a concrete adapter implementation for testing."""

        class ConcreteAdapter(RuntimeAdapter):
            """Minimal concrete implementation for testing base class."""

            def load(self, model_path: Path) -> None:
                self.model = MagicMock()  # type: ignore[assignment]

            def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return {"output": np.array([1.0])}

            @property
            def input_names(self) -> list[str]:
                return ["input"]

            @property
            def output_names(self) -> list[str]:
                return ["output"]

        return ConcreteAdapter

    def test_base_init(self, concrete_adapter_class) -> None:
        """Test base adapter initialization with device and kwargs."""
        adapter = concrete_adapter_class(device="cpu", custom_param="value")
        assert adapter.device == "cpu"
        assert adapter.config == {"custom_param": "value"}
        assert adapter.model is None

    def test_base_repr(self, concrete_adapter_class) -> None:
        """Test base adapter string representation."""
        adapter = concrete_adapter_class(device="gpu")
        assert "ConcreteAdapter" in repr(adapter)
        assert "gpu" in repr(adapter)

    def test_abstract_methods_required(self) -> None:
        """Test that abstract methods must be implemented."""

        class IncompleteAdapter(RuntimeAdapter):
            def load(self, model_path: Path) -> None:
                raise NotImplementedError

            def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                raise NotImplementedError

            @property
            def input_names(self) -> list[str]:
                raise NotImplementedError

            @property
            def output_names(self) -> list[str]:
                raise NotImplementedError

        assert inspect.isabstract(RuntimeAdapter)

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference adapters."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from physicalai.data.observation import Observation
from physicalai.export.backends import TorchExportParameters
from physicalai.export.mixin_policy import ExportBackend
from physicalai.inference.adapters import (
    ExecuTorchAdapter,
    ONNXAdapter,
    OpenVINOAdapter,
    RuntimeAdapter,
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
        mock_compiled_model = MagicMock()
        mock_input, mock_output = Mock(), Mock()
        mock_input.any_name, mock_output.any_name = "input", "output"
        mock_model.inputs = [mock_input]
        mock_compiled_model.outputs = [mock_output]
        mock_compiled_model.return_value = [np.array([[1.0, 2.0]])]

        mock_ov = MagicMock()
        mock_ov.Core.return_value.read_model.return_value = mock_model
        mock_ov.Core.return_value.compile_model.return_value = mock_compiled_model

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

    @staticmethod
    def _write_policy_metadata(tmp_path: Path) -> Path:
        model_path = tmp_path / "model.pt"
        metadata_path = tmp_path / "metadata.yaml"
        model_path.touch()
        with metadata_path.open("w") as f:
            f.write("policy_class: physicalai.policies.act.ACT\n")
        return model_path

    def test_lifecycle(self, tmp_path: Path) -> None:
        """Test complete adapter lifecycle: init, load, predict with numpy inputs."""
        model_path = self._write_policy_metadata(tmp_path)

        mock_model = MagicMock()
        # Policy forward returns a tensor (action output)
        mock_model.return_value = torch.tensor([[1.0, 2.0]])
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.extra_export_args = {"torch": TorchExportParameters()}

        with patch("physicalai.policies.act.ACT.load_from_checkpoint", return_value=mock_model):
            adapter = TorchAdapter(device="cpu")
            assert adapter.device == torch.device("cpu")
            assert "cpu" in repr(adapter)

            adapter.load(model_path)
            # Torch adapter intentionally keeps input_names empty and parses
            # structured payloads in predict() via Observation.from_dict(...).
            assert adapter.input_names == []
            assert adapter.output_names == ["action"]

            # Predict with dict[str, np.ndarray] — same contract as all adapters
            outputs = adapter.predict({
                "state": np.array([[0.5, 0.3]], dtype=np.float32),
                "images": np.random.rand(1, 3, 96, 96).astype(np.float32),
            })
            assert "action" in outputs
            assert isinstance(outputs["action"], np.ndarray)
            np.testing.assert_array_almost_equal(outputs["action"], [[1.0, 2.0]])

    def test_predict_with_nested_images(self, tmp_path: Path) -> None:
        """Test predict with multi-camera images (dict of numpy arrays)."""
        model_path = self._write_policy_metadata(tmp_path)

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.2]])
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.model.sample_input = {"state": torch.zeros(1, 2), "images": torch.zeros(1, 3, 96, 96)}
        mock_model.extra_export_args = {"torch": TorchExportParameters()}

        with patch("physicalai.policies.act.ACT.load_from_checkpoint", return_value=mock_model):
            adapter = TorchAdapter(device="cpu")
            adapter.load(model_path)

            # Multi-camera images as a nested dict
            outputs = adapter.predict({
                "state": np.array([[1.0, 2.0]], dtype=np.float32),
                "images": {
                    "top": np.random.rand(1, 3, 96, 96).astype(np.float32),
                    "front": np.random.rand(1, 3, 96, 96).astype(np.float32),
                },
            })
            assert "action" in outputs
            assert isinstance(outputs["action"], np.ndarray)

    def test_load_with_missing_sample_input_keeps_empty_input_names(self, tmp_path: Path) -> None:
        """Test missing sample_input still keeps empty input_names for torch adapter."""
        model_path = self._write_policy_metadata(tmp_path)

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.2]])
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        if hasattr(mock_model.model, "sample_input"):
            del mock_model.model.sample_input
        mock_model.extra_export_args = {"torch": TorchExportParameters()}

        with patch("physicalai.policies.act.ACT.load_from_checkpoint", return_value=mock_model):
            adapter = TorchAdapter(device="cpu")
            adapter.load(model_path)

            assert adapter.input_names == []
            assert adapter.output_names == ["action"]

    def test_observation_from_numpy_inputs(self) -> None:
        """Test that numpy dict inputs are correctly converted to an Observation with torch tensors."""
        inputs = {
            "state": np.array([[1.0, 2.0]], dtype=np.float32),
            "images": {
                "top": np.array([[[[0.5]]]], dtype=np.float32),
            },
        }
        obs = Observation.from_dict(inputs).to_torch("cpu")

        assert isinstance(obs.state, torch.Tensor)
        assert isinstance(obs.images["top"], torch.Tensor)
        torch.testing.assert_close(obs.state, torch.tensor([[1.0, 2.0]]))

    def test_error_cases(self, tmp_path: Path) -> None:
        """Test error handling for file not found and predict without load."""
        adapter = TorchAdapter()

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            adapter.load(Path("/nonexistent/model.pt"))

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.predict({"input": np.array([1.0])})

    def test_load_failure(self, tmp_path: Path) -> None:
        """Test error handling when torch.load fails."""
        model_path = self._write_policy_metadata(tmp_path)

        with patch("physicalai.policies.act.ACT.load_from_checkpoint", side_effect=RuntimeError("Load error")):
            adapter = TorchAdapter()
            with pytest.raises(RuntimeError, match="Failed to load"):
                adapter.load(model_path)

    def test_device_selection(self) -> None:
        """Test device selection for Torch adapter."""
        adapter_cpu = TorchAdapter(device="cpu")
        assert adapter_cpu.device == torch.device("cpu")

        adapter_cuda = TorchAdapter(device="cuda")
        assert adapter_cuda.device == torch.device("cuda")

    def test_input_output_names_before_load(self) -> None:
        """Test input/output names return empty lists before model is loaded."""
        adapter = TorchAdapter()
        adapter._policy = MagicMock()

        assert adapter.input_names == []
        assert adapter.output_names == []


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
        with pytest.raises(TypeError):
            RuntimeAdapter()  # type: ignore[abstract]


class TestDefaultDevice:
    """Test default_device() method for all adapters."""

    def test_runtime_adapter_default_device(self) -> None:
        """Test RuntimeAdapter base class default_device returns 'cpu'."""

        class ConcreteAdapter(RuntimeAdapter):
            def load(self, model_path: Path) -> None:
                pass

            def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return {"output": np.array([1.0])}

            @property
            def input_names(self) -> list[str]:
                return ["input"]

            @property
            def output_names(self) -> list[str]:
                return ["output"]

        adapter = ConcreteAdapter()
        assert adapter.default_device() == "cpu"

    def test_onnx_adapter_default_device(self) -> None:
        """Test ONNXAdapter default_device returns a string."""
        adapter = ONNXAdapter()
        result = adapter.default_device()
        assert isinstance(result, str)

    def test_openvino_adapter_default_device(self) -> None:
        """Test OpenVINOAdapter default_device returns 'CPU'."""
        adapter = OpenVINOAdapter()
        assert adapter.default_device() == "CPU"


class TestExecuTorchAdapter:
    """Test ExecuTorch inference adapter."""

    @staticmethod
    def _build_executorch_mocks() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        """Build the mock objects for executorch.runtime.

        Returns:
            Tuple of (mock_et_runtime, mock_runtime_instance, mock_program, mock_method).
        """
        mock_method = MagicMock()
        mock_program = MagicMock()
        mock_program.load_method.return_value = mock_method

        mock_runtime_instance = MagicMock()
        mock_runtime_instance.load_program.return_value = mock_program

        mock_runtime_class = MagicMock()
        mock_runtime_class.get.return_value = mock_runtime_instance

        mock_et_runtime = MagicMock()
        mock_et_runtime.Runtime = mock_runtime_class

        return mock_et_runtime, mock_runtime_instance, mock_program, mock_method

    def test_load_happy_path(self, tmp_path: Path) -> None:
        """Test successful load with metadata."""
        model_path = tmp_path / "model.pte"
        model_path.touch()

        metadata_path = tmp_path / "metadata.yaml"
        metadata_path.write_text("input_names: [state, action]\noutput_names: [prediction]\n")

        mock_et_runtime, _, _, _ = self._build_executorch_mocks()

        with patch.dict("sys.modules", {"executorch": MagicMock(), "executorch.runtime": mock_et_runtime}):
            adapter = ExecuTorchAdapter()
            adapter.load(model_path)

            assert adapter.input_names == ["state", "action"]
            assert adapter.output_names == ["prediction"]

    def test_load_file_not_found(self) -> None:
        """Test FileNotFoundError for missing file."""
        adapter = ExecuTorchAdapter()
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            adapter.load(Path("/nonexistent/model.pte"))

    def test_load_import_error(self, tmp_path: Path) -> None:
        """Test ImportError when executorch not installed."""
        model_path = tmp_path / "model.pte"
        model_path.touch()

        with patch.dict("sys.modules", {"executorch": None, "executorch.runtime": None}):
            adapter = ExecuTorchAdapter()
            with pytest.raises(ImportError, match="executorch"):
                adapter.load(model_path)

    def test_predict_happy_path(self, tmp_path: Path) -> None:
        """Test successful prediction."""
        model_path = tmp_path / "model.pte"
        model_path.touch()

        metadata_path = tmp_path / "metadata.yaml"
        metadata_path.write_text("input_names: [state]\noutput_names: [action]\n")

        mock_et_runtime, _, _, mock_method = self._build_executorch_mocks()
        mock_method.execute.return_value = [torch.tensor([[1.0, 2.0]])]

        with patch.dict("sys.modules", {"executorch": MagicMock(), "executorch.runtime": mock_et_runtime}):
            adapter = ExecuTorchAdapter()
            adapter.load(model_path)

            outputs = adapter.predict({"state": np.array([[0.5]])})
            assert "action" in outputs
            assert isinstance(outputs["action"], np.ndarray)
            np.testing.assert_array_almost_equal(outputs["action"], [[1.0, 2.0]])

    def test_predict_not_loaded(self) -> None:
        """Test RuntimeError when predict called before load."""
        adapter = ExecuTorchAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.predict({"state": np.array([[0.5]])})

    def test_predict_dict_to_tuple_ordering(self) -> None:
        """Test input order matches input_names, not dict insertion order."""
        adapter = ExecuTorchAdapter()
        mock_method = MagicMock()
        mock_method.execute.return_value = [torch.tensor([[0.0]])]

        adapter._method = mock_method
        adapter._input_names = ["b_input", "a_input"]
        adapter._output_names = ["out"]

        adapter.predict({"a_input": np.array([[1.0]]), "b_input": np.array([[2.0]])})

        # Verify execute was called with ordered inputs matching input_names order
        call_args = mock_method.execute.call_args[0][0]
        # First tensor should be b_input (2.0), second should be a_input (1.0)
        np.testing.assert_array_almost_equal(call_args[0].numpy(), [[2.0]])
        np.testing.assert_array_almost_equal(call_args[1].numpy(), [[1.0]])

    def test_input_output_names(self) -> None:
        """Test properties return correct values."""
        adapter = ExecuTorchAdapter()
        adapter._input_names = ["x"]
        adapter._output_names = ["y"]

        assert adapter.input_names == ["x"]
        assert adapter.output_names == ["y"]

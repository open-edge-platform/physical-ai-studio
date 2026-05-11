# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference adapters shipped with the core ``physicalai`` package.

Covers the ``onnx`` and ``openvino`` adapters, the :class:`RuntimeAdapter`
base class, and the :func:`get_adapter` factory for those backends.

Tests for the ``torch`` and ``executorch`` adapters live alongside the
distribution that contributes them (``library/tests/unit/inference``).
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from physicalai.inference.adapters import (
    ONNXAdapter,
    OpenVINOAdapter,
    RuntimeAdapter,
    get_adapter,
)


class TestGetAdapter:
    """Test adapter factory function."""

    @pytest.mark.parametrize(
        ("backend_name", "expected_type"),
        [
            ("openvino", OpenVINOAdapter),
            ("onnx", ONNXAdapter),
        ],
    )
    def test_get_adapter(
        self,
        backend_name: str,
        expected_type: type[RuntimeAdapter],
    ) -> None:
        """Test get_adapter returns correct type for a string backend name."""
        adapter = get_adapter(backend_name)
        assert isinstance(adapter, expected_type)

    def test_invalid_backend(self) -> None:
        """Test invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="No adapter registered for backend"):
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
    """Test default_device() method for core adapters."""

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

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for inference adapters contributed by ``physicalai-train``.

Covers the ``torch`` and ``executorch`` adapters that this distribution
registers via the ``physicalai.inference.adapters`` entry-point group.

Tests for the core ``onnx`` and ``openvino`` adapters live in
``physicalai/tests/unit/inference/test_adapters.py``.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from physicalai.data.observation import Observation
from physicalai.export.backends import TorchExportParameters
from physicalai.export.mixin_policy import ExportBackend
from physicalai.inference.adapters import get_adapter
from physicalai.inference.adapters.executorch import ExecuTorchAdapter
from physicalai.inference.adapters.pytorch import TorchAdapter


class TestGetAdapter:
    """Test adapter factory function for adapters contributed here."""

    @pytest.mark.parametrize("backend_type", [ExportBackend, str])
    def test_get_torch_adapter(self, backend_type: type) -> None:
        """Test get_adapter returns TorchAdapter for both enum and string inputs."""
        backend = backend_type("torch") if backend_type == ExportBackend else "torch"
        adapter = get_adapter(backend)
        assert isinstance(adapter, TorchAdapter)


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
            assert adapter.device == "cpu"
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
        assert adapter_cpu.device == "cpu"

        adapter_cuda = TorchAdapter(device="cuda")
        assert adapter_cuda.device == "cuda"

    def test_input_output_names_before_load(self) -> None:
        """Test input/output names return empty lists before model is loaded."""
        adapter = TorchAdapter()
        adapter._policy = MagicMock()

        assert adapter.input_names == []
        assert adapter.output_names == []


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

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InferenceModel."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from physicalai.export.mixin_export import ExportBackend
from physicalai.inference.adapters import RuntimeAdapter
from physicalai.inference.model import InferenceModel


class TestAdapter(RuntimeAdapter):
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._policy: torch.nn.Module | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path | str) -> None:
        pass

    def predict(self, inputs: dict[str, np.ndarray]):
        pass

    @property
    def input_names(self) -> list[str]:
        return []

    @property
    def output_names(self) -> list[str]:
        return []


def test_exported_metadata_controls_action_queue(
    tmp_path: Path,
    mock_adapter: MagicMock,
    sample_observation: dict[str, np.ndarray],
) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    import yaml

    metadata = {
        "policy_class": "physicalai.policies.lerobot.smolvla.SmolVLA",
        "backend": "openvino",
        "use_action_queue": True,
        "chunk_size": 3,
    }

    with (export_dir / "metadata.yaml").open("w") as f:
        yaml.dump(metadata, f)

    (export_dir / "smolvla.xml").touch()
    (export_dir / "smolvla.bin").touch()

    with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
        model = InferenceModel(export_dir)

        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 3, 2)}

        action1 = model.select_action(sample_observation)
        action2 = model.select_action(sample_observation)

        assert action1.shape == (1, 2)
        assert action2.shape == (1, 2)
        assert mock_adapter.predict.call_count == 1


@pytest.fixture
def mock_export_dir(tmp_path: Path) -> Path:
    """Create mock export directory with metadata."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    # Create metadata
    metadata = {
        "policy_class": "physicalai.policies.act.ACT",
        "backend": "openvino",
        "use_action_queue": True,
        "chunk_size": 10,
    }
    import yaml

    with (export_dir / "metadata.yaml").open("w") as f:
        yaml.dump(metadata, f)

    # Create dummy model file
    (export_dir / "act.xml").touch()
    (export_dir / "act.bin").touch()

    return export_dir


@pytest.fixture
def mock_adapter():
    """Create mock adapter for testing."""
    adapter = MagicMock()
    adapter.input_names = ["state", "images"]
    adapter.output_names = ["actions"]
    adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
    adapter.default_device.return_value = "cpu"
    return adapter


@pytest.fixture
def sample_observation() -> dict[str, np.ndarray]:
    """Create sample observation dict for testing."""
    return {
        "state": np.random.randn(1, 4).astype(np.float32),
        "images": np.random.randn(1, 3, 224, 224).astype(np.float32),
    }


class TestInferenceModelInit:
    """Test InferenceModel initialization and auto-detection."""

    def test_init_with_valid_directory(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test initialization with valid export directory."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            assert model.export_dir == mock_export_dir
            assert model.policy_name == "act"
            assert model.backend == ExportBackend.OPENVINO
            assert model.use_action_queue is True
            assert model.chunk_size == 10

    def test_init_with_nonexistent_directory(self) -> None:
        """Test initialization fails with nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Export directory not found"):
            InferenceModel("/nonexistent/path")

    @pytest.mark.parametrize(
        ("backend_str", "expected", "file_ext"),
        [
            ("openvino", ExportBackend.OPENVINO, ".xml"),
            ("onnx", ExportBackend.ONNX, ".onnx"),
            ("torch_export_ir", ExportBackend.TORCH_EXPORT_IR, ".pt2"),
        ],
    )
    def test_init_with_explicit_backend(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        backend_str: str,
        expected: ExportBackend,
        file_ext: str,
    ) -> None:
        """Test initialization with explicit backend specification."""
        # Create export dir with appropriate model file
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / f"model{file_ext}").touch()

        import yaml

        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": "physicalai.policies.act.ACT"}, f)

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir, backend=backend_str)
            assert model.backend == expected

    def test_load_classmethod(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test convenience load() classmethod."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel.load(mock_export_dir)
            assert isinstance(model, InferenceModel)
            assert model.policy_name == "act"


class TestMetadataLoading:
    """Test metadata loading from different formats."""

    @pytest.mark.parametrize(
        ("format_type", "metadata_content"),
        [
            ("yaml", {"policy_class": "physicalai.policies.dummy.Dummy", "chunk_size": 5}),
            ("json", {"policy_class": "physicalai.policies.act.ACT", "backend": "onnx"}),
        ],
    )
    def test_load_metadata_formats(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        format_type: str,
        metadata_content: dict,
    ) -> None:
        """Test loading metadata from YAML and JSON formats."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()

        if format_type == "yaml":
            import yaml

            with (export_dir / "metadata.yaml").open("w") as f:
                yaml.dump(metadata_content, f)
            (export_dir / "dummy.xml").touch()
        else:
            import json

            with (export_dir / "metadata.json").open("w") as f:
                json.dump(metadata_content, f)
            (export_dir / "act.onnx").touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir)
            assert model.metadata == metadata_content

    def test_no_metadata_fallback(self, tmp_path: Path, mock_adapter: MagicMock) -> None:
        """Test graceful handling when no metadata file exists."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / "model.onnx").touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            assert InferenceModel(export_dir).metadata == {}


class TestAutoDetection:
    """Test auto-detection of policy name, backend, and device."""

    @pytest.mark.parametrize(
        ("policy_class", "expected_name"),
        [
            ("physicalai.policies.act.ACT", "act"),
            ("physicalai.policies.diffusion.Diffusion", "diffusion"),
            ("physicalai.policies.dummy.Dummy", "dummy"),
        ],
    )
    def test_policy_detection(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        policy_class: str,
        expected_name: str,
    ) -> None:
        """Test policy name detection from metadata."""
        import yaml

        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": policy_class}, f)
        (export_dir / f"{expected_name}.xml").touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            assert InferenceModel(export_dir).policy_name == expected_name

    @pytest.mark.parametrize(
        ("file_ext", "expected_backend"),
        [(".xml", ExportBackend.OPENVINO), (".onnx", ExportBackend.ONNX), (".pt2", ExportBackend.TORCH_EXPORT_IR)],
    )
    def test_backend_detection(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        file_ext: str,
        expected_backend: ExportBackend,
    ) -> None:
        """Test backend detection from file extensions."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / f"model{file_ext}").touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            assert InferenceModel(export_dir).backend == expected_backend

    @pytest.mark.parametrize("cuda_available", [True, False])
    @pytest.mark.parametrize(
        ("backend_file", "backend_type"),
        [
            ("model.xml", "openvino"),
            ("model.onnx", "onnx"),
            ("model.pt2", "torch_export_ir"),
        ],
    )
    def test_device_detection(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        backend_file: str,
        backend_type: str,
        cuda_available: bool,
    ) -> None:
        """Test device detection based on backend and CUDA availability."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / backend_file).touch()

        # OpenVINO always uses CPU, others use cuda/cpu based on availability
        if backend_type == "openvino":
            expected_device = "CPU"
        else:
            expected_device = "cuda" if cuda_available else "cpu"

        # Configure mock to return the expected device
        mock_adapter.default_device.return_value = expected_device

        with patch("torch.cuda.is_available", return_value=cuda_available):
            with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
                assert InferenceModel(export_dir, device="auto").device == expected_device

    def test_device_setting(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test manual device setting."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        backend_file = "model.onnx"
        (export_dir / backend_file).touch()

        expected_device = "xpu"
        with patch("physicalai.inference.model.get_adapter", return_value=TestAdapter(device=expected_device)):
            assert InferenceModel(export_dir, device=expected_device).device == expected_device


class TestSelectAction:
    """Test action selection with different configurations."""

    def test_select_action_no_queue(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test action selection without action queue."""
        # Configure for non-chunked policy
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model.use_action_queue = False
            model.chunk_size = 1

            # Mock adapter returns (1, 1, 2) - remove temporal dim
            mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}

            action = model.select_action(sample_observation)

            assert isinstance(action, np.ndarray)
            assert action.shape == (1, 2)  # (batch, action_dim)
            mock_adapter.predict.assert_called_once()

    def test_select_action_with_queue(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test action selection with action queue for chunked policy."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model.use_action_queue = True
            model.chunk_size = 10

            # Mock returns chunk of 10 actions: (batch=1, chunk=10, action_dim=2)
            mock_adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}

            # First call should populate queue
            action1 = model.select_action(sample_observation)
            assert action1.shape == (1, 2)
            assert len(model._action_queue) == 9  # 10 - 1 returned

            # Next calls should use queue without inference
            action2 = model.select_action(sample_observation)
            assert action2.shape == (1, 2)
            assert len(model._action_queue) == 8
            mock_adapter.predict.assert_called_once()  # Still only called once

    def test_select_action_queue_refill(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        """Test action queue refills when empty."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model.use_action_queue = True
            model.chunk_size = 3

            mock_adapter.predict.return_value = {"actions": np.random.randn(1, 3, 2)}

            # Exhaust queue
            for _ in range(3):
                model.select_action(sample_observation)

            assert len(model._action_queue) == 0
            mock_adapter.predict.assert_called_once()

            # Next call should refill
            model.select_action(sample_observation)
            assert len(model._action_queue) == 2
            assert mock_adapter.predict.call_count == 2

    def test_reset_clears_queue(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test reset() clears action queue."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model._action_queue.extend([np.random.randn(1, 2).astype(np.float32) for _ in range(5)])

            model.reset()
            assert len(model._action_queue) == 0

    def test_select_action_with_numpy_dict_input(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test select_action with dict[str, np.ndarray] returns np.ndarray (no backward compat wrap)."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model.use_action_queue = False
            model.chunk_size = 1

            # Mock adapter returns (1, 1, 2) - remove temporal dim
            mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}

            # Pass dict[str, np.ndarray] directly
            numpy_input = {"state": np.random.randn(1, 3).astype(np.float32),
                           "images": np.random.randn(1, 3, 224, 224).astype(np.float32)}
            action = model.select_action(numpy_input)

            assert isinstance(action, np.ndarray)
            assert action.shape == (1, 2)  # (batch, action_dim)
            mock_adapter.predict.assert_called_once()


class TestInputPreparation:
    """Test observation-to-input filtering."""

    def test_prepare_inputs_filters_to_adapter_input_names(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test _prepare_inputs filters observation to adapter's expected input names."""
        mock_adapter.input_names = ["state", "images"]
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            inputs = {
                "state": np.random.randn(1, 4).astype(np.float32),
                "images": np.random.randn(1, 3, 224, 224).astype(np.float32),
                "action": np.random.randn(1, 2).astype(np.float32),
                "episode_index": np.array([0]),
                "task": np.array([1]),
            }

            result = model._prepare_inputs(inputs)

            assert set(result.keys()) == {"state", "images"}
            np.testing.assert_array_equal(result["state"], inputs["state"])
            np.testing.assert_array_equal(result["images"], inputs["images"])
            assert "action" not in result
            assert "episode_index" not in result
            assert "task" not in result

    def test_prepare_inputs_passthrough_when_no_input_names(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test _prepare_inputs passes through when adapter has no input names."""
        mock_adapter.input_names = []
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            inputs = {
                "state": np.random.randn(1, 4).astype(np.float32),
                "extra": np.random.randn(1, 2).astype(np.float32),
            }

            result = model._prepare_inputs(inputs)

            assert result is inputs

    def test_prepare_inputs_only_matching_keys(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test _prepare_inputs returns only keys that exist in both observation and input_names."""
        mock_adapter.input_names = ["state", "images", "extra_input"]
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            inputs = {
                "state": np.random.randn(1, 4).astype(np.float32),
                "images": np.random.randn(1, 3, 224, 224).astype(np.float32),
                "extra_input": np.random.randn(1, 2).astype(np.float32),
                "unrelated": np.random.randn(1, 2).astype(np.float32),
            }

            result = model._prepare_inputs(inputs)

            assert set(result.keys()) == {"state", "images", "extra_input"}
            assert "unrelated" not in result

    def test_prepare_inputs_nested_payload_handling(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test nested key handling for dotted adapter inputs."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            mock_adapter.input_names = ["state", "images.top"]
            inputs = {
                "state": np.random.randn(1, 4).astype(np.float32),
                "images": {
                    "top": np.random.randn(1, 3, 224, 224).astype(np.float32),
                },
                "action": np.random.randn(1, 2).astype(np.float32),
            }

            result = model._prepare_inputs(inputs)
            assert set(result.keys()) == {"state", "images.top"}
            assert "action" not in result


class TestActionOutputKey:
    """Test action output key detection."""

    @pytest.mark.parametrize(
        ("outputs", "expected_key"),
        [
            ({"actions": np.array([1, 2])}, "actions"),
            ({"action": np.array([1, 2])}, "action"),
            ({"output": np.array([1, 2])}, "output"),
            ({"pred_actions": np.array([1, 2])}, "pred_actions"),
            ({"custom_key": np.array([1, 2])}, "custom_key"),  # Fallback to first
        ],
    )
    def test_get_action_output_key(self, outputs: dict[str, np.ndarray], expected_key: str) -> None:
        """Test action output key detection with different naming."""
        key = InferenceModel._get_action_output_key(outputs)
        assert key == expected_key


class TestModelPathResolution:
    """Test model file path resolution."""

    @pytest.mark.parametrize(
        ("backend", "file_ext"),
        [
            (ExportBackend.OPENVINO, ".xml"),
            (ExportBackend.ONNX, ".onnx"),
            (ExportBackend.TORCH_EXPORT_IR, ".pt2"),
        ],
    )
    def test_get_model_path_with_policy_name(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        backend: ExportBackend,
        file_ext: str,
    ) -> None:
        """Test model path resolution with policy name."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        model_file = export_dir / f"act{file_ext}"
        model_file.touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir, policy_name="act", backend=backend)
            path = model._get_model_path()
            assert path == model_file

    def test_get_model_path_without_policy_name(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test model path resolution without policy name."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        model_file = export_dir / "model.onnx"
        model_file.touch()

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir, policy_name=None, backend="onnx")
            path = model._get_model_path()
            assert path == model_file

    def test_get_model_path_not_found(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test model path resolution raises when file not found."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()

        # Add metadata so policy name detection works
        import yaml

        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": "physicalai.policies.act.ACT"}, f)

        with (
            patch("physicalai.inference.model.get_adapter", return_value=mock_adapter),
            pytest.raises(FileNotFoundError, match="No .* model file found"),
        ):
            model = InferenceModel(export_dir, backend="onnx")
            model._get_model_path()


class TestRepr:
    """Test string representation."""

    def test_repr(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test __repr__ returns informative string."""
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            repr_str = repr(model)

            assert "InferenceModel" in repr_str
            assert "act" in repr_str
            assert "openvino" in repr_str
            assert model.device in repr_str

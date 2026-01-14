# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InferenceModel."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from getiaction.data.observation import Observation
from getiaction.export.mixin_export import ExportBackend
from getiaction.inference.model import InferenceModel


def test_exported_metadata_controls_action_queue(
    tmp_path: Path, mock_adapter: MagicMock, sample_observation: Observation
) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    import yaml

    metadata = {
        "policy_class": "getiaction.policies.lerobot.smolvla.SmolVLA",
        "backend": "openvino",
        "use_action_queue": True,
        "chunk_size": 3,
    }

    with (export_dir / "metadata.yaml").open("w") as f:
        yaml.dump(metadata, f)

    (export_dir / "smolvla.xml").touch()
    (export_dir / "smolvla.bin").touch()

    with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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
        "policy_class": "getiaction.policies.act.ACT",
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
    adapter.input_names = ["observation.state", "observation.image"]
    adapter.output_names = ["actions"]
    adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
    return adapter


@pytest.fixture
def sample_observation() -> Observation:
    """Create sample observation for testing."""
    return Observation(
        state=torch.randn(1, 4),
        images=torch.randn(1, 3, 224, 224),
    )


class TestInferenceModelInit:
    """Test InferenceModel initialization and auto-detection."""

    def test_init_with_valid_directory(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test initialization with valid export directory."""
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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
            yaml.dump({"policy_class": "getiaction.policies.act.ACT"}, f)

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir, backend=backend_str)
            assert model.backend == expected

    def test_load_classmethod(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test convenience load() classmethod."""
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel.load(mock_export_dir)
            assert isinstance(model, InferenceModel)
            assert model.policy_name == "act"


class TestMetadataLoading:
    """Test metadata loading from different formats."""

    @pytest.mark.parametrize(
        ("format_type", "metadata_content"),
        [
            ("yaml", {"policy_class": "getiaction.policies.dummy.Dummy", "chunk_size": 5}),
            ("json", {"policy_class": "getiaction.policies.act.ACT", "backend": "onnx"}),
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

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(export_dir)
            assert model.metadata == metadata_content

    def test_no_metadata_fallback(self, tmp_path: Path, mock_adapter: MagicMock) -> None:
        """Test graceful handling when no metadata file exists."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / "model.onnx").touch()

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            assert InferenceModel(export_dir).metadata == {}


class TestAutoDetection:
    """Test auto-detection of policy name, backend, and device."""

    @pytest.mark.parametrize(
        ("policy_class", "expected_name"),
        [
            ("getiaction.policies.act.ACT", "act"),
            ("getiaction.policies.diffusion.Diffusion", "diffusion"),
            ("getiaction.policies.dummy.Dummy", "dummy"),
        ],
    )
    def test_policy_detection(self, tmp_path: Path, mock_adapter: MagicMock, policy_class: str, expected_name: str) -> None:
        """Test policy name detection from metadata."""
        import yaml

        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": policy_class}, f)
        (export_dir / f"{expected_name}.xml").touch()

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            assert InferenceModel(export_dir).policy_name == expected_name

    @pytest.mark.parametrize(
        ("file_ext", "expected_backend"),
        [(".xml", ExportBackend.OPENVINO), (".onnx", ExportBackend.ONNX), (".pt2", ExportBackend.TORCH_EXPORT_IR)],
    )
    def test_backend_detection(self, tmp_path: Path, mock_adapter: MagicMock, file_ext: str, expected_backend: ExportBackend) -> None:
        """Test backend detection from file extensions."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / f"model{file_ext}").touch()

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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

        with patch("torch.cuda.is_available", return_value=cuda_available):
            with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
                assert InferenceModel(export_dir, device="auto").device == expected_device


class TestSelectAction:
    """Test action selection with different configurations."""

    def test_select_action_no_queue(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
        sample_observation: Observation,
    ) -> None:
        """Test action selection without action queue."""
        # Configure for non-chunked policy
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model.use_action_queue = False
            model.chunk_size = 1

            # Mock adapter returns (1, 1, 2) - remove temporal dim
            mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}

            action = model.select_action(sample_observation)

            assert isinstance(action, torch.Tensor)
            assert action.shape == (1, 2)  # (batch, action_dim)
            mock_adapter.predict.assert_called_once()

    def test_select_action_with_queue(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
        sample_observation: Observation,
    ) -> None:
        """Test action selection with action queue for chunked policy."""
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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
        sample_observation: Observation,
    ) -> None:
        """Test action queue refills when empty."""
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            model._action_queue.extend([torch.randn(1, 2) for _ in range(5)])

            model.reset()
            assert len(model._action_queue) == 0


class TestInputPreparation:
    """Test observation-to-input conversion."""

    def test_prepare_inputs_first_party(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test input preparation with first-party naming (state, images)."""
        mock_adapter.input_names = ["state", "images"]

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            obs = Observation(
                state=torch.randn(1, 4),
                images=torch.randn(1, 3, 224, 224),
            )

            inputs = model._prepare_inputs(obs)

            assert "state" in inputs
            assert "images" in inputs
            assert isinstance(inputs["state"], np.ndarray)
            assert isinstance(inputs["images"], np.ndarray)

    def test_prepare_inputs_lerobot(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test input preparation with LeRobot naming (observation.state, observation.image)."""
        mock_adapter.input_names = ["observation.state", "observation.image"]

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            obs = Observation(
                state=torch.randn(1, 4),
                images=torch.randn(1, 3, 224, 224),
            )

            inputs = model._prepare_inputs(obs)

            assert "observation.state" in inputs
            assert "observation.image" in inputs

    def test_prepare_inputs_handles_none_values(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test input preparation skips None values."""
        mock_adapter.input_names = ["state", "images"]

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            obs = Observation(state=torch.randn(1, 4), images=torch.randn(1, 3, 224, 224), action=None)
            inputs = model._prepare_inputs(obs)

            assert "state" in inputs
            assert "images" in inputs
            assert "action" not in inputs

    def test_prepare_inputs_numpy_passthrough(
        self,
        mock_export_dir: Path,
        mock_adapter: MagicMock,
    ) -> None:
        """Test input preparation handles numpy arrays directly."""
        mock_adapter.input_names = ["state"]

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)

            state_np = np.random.randn(1, 4)
            obs = Observation(state=state_np, images=None)
            inputs = model._prepare_inputs(obs)

            assert np.array_equal(inputs["state"], state_np)


class TestFieldMapping:
    """Test field mapping between observation and model inputs."""

    @pytest.mark.parametrize(
        ("obs_fields", "model_inputs", "expected"),
        [
            # First-party naming
            ({"state": 1}, {"state"}, {"state": "state"}),
            ({"images": 1}, {"images"}, {"images": "images"}),
            # LeRobot naming
            ({"state": 1}, {"observation.state"}, {"state": "observation.state"}),
            ({"images": 1}, {"observation.image"}, {"images": "observation.image"}),
            # Mixed
            (
                {"state": 1, "images": 1},
                {"state", "observation.image"},
                {"state": "state", "images": "observation.image"},
            ),
            # No match
            ({"unknown": 1}, {"state"}, {}),
        ],
    )
    def test_build_field_mapping(
        self,
        obs_fields: dict[str, int],
        model_inputs: set[str],
        expected: dict[str, str],
    ) -> None:
        """Test field mapping handles different naming conventions."""
        mapping = InferenceModel._build_field_mapping(obs_fields, model_inputs)
        assert mapping == expected

    def test_exact_match_first_party(self) -> None:
        """Test exact matches for first-party naming convention."""
        obs_dict = {
            "state": np.array([1.0, 2.0]),
            "images": [np.random.randn(3, 224, 224)],
        }
        expected_inputs = {"state", "images"}

        mapping = InferenceModel._build_field_mapping(obs_dict, expected_inputs)

        assert mapping == {"state": "state", "images": "images"}

    def test_exact_match_lerobot(self) -> None:
        """Test exact matches for LeRobot naming convention."""
        obs_dict = {
            "state": np.array([1.0, 2.0]),
            "images": [np.random.randn(3, 224, 224)],
        }
        expected_inputs = {"observation.state", "observation.image"}

        mapping = InferenceModel._build_field_mapping(obs_dict, expected_inputs)

        assert mapping == {
            "state": "observation.state",
            "images": "observation.image",
        }


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

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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

        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
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
            yaml.dump({"policy_class": "getiaction.policies.act.ACT"}, f)

        with (
            patch("getiaction.inference.model.get_adapter", return_value=mock_adapter),
            pytest.raises(FileNotFoundError, match="No .* model file found"),
        ):
            model = InferenceModel(export_dir, backend="onnx")
            model._get_model_path()


class TestRepr:
    """Test string representation."""

    def test_repr(self, mock_export_dir: Path, mock_adapter: MagicMock) -> None:
        """Test __repr__ returns informative string."""
        with patch("getiaction.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir)
            repr_str = repr(model)

            assert "InferenceModel" in repr_str
            assert "act" in repr_str
            assert "openvino" in repr_str
            assert model.device in repr_str

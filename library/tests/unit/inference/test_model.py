# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for InferenceModel and inference runners."""

import json
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from physicalai.export.mixin_policy import ExportBackend
from physicalai.inference.adapters import RuntimeAdapter
from physicalai.inference.model import InferenceModel
from physicalai.inference.postprocessors.base import Postprocessor
from physicalai.inference.preprocessors.base import Preprocessor
from physicalai.inference.runners import (
    ActionChunking,
    InferenceRunner,
    SinglePass,
    get_runner,
)

class TestAdapter(RuntimeAdapter):
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._policy: torch.nn.Module | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    def load(self, model_path: Path | str) -> None:
        pass

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {}

    @property
    def input_names(self) -> list[str]:
        return []

    @property
    def output_names(self) -> list[str]:
        return []


def _make_legacy_export_dir(
    tmp_path: Path,
    *,
    use_action_queue: bool = True,
    chunk_size: int = 10,
    policy_class: str = "physicalai.policies.act.ACT",
    backend: str = "openvino",
) -> Path:
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    metadata = {
        "policy_class": policy_class,
        "backend": backend,
        "use_action_queue": use_action_queue,
        "chunk_size": chunk_size,
    }
    with (export_dir / "metadata.yaml").open("w") as f:
        yaml.dump(metadata, f)
    (export_dir / "act.xml").touch()
    (export_dir / "act.bin").touch()
    return export_dir


def _make_manifest_export_dir(
    tmp_path: Path,
    *,
    kind: str = "action_chunking",
    chunk_size: int = 10,
    backend: str = "openvino",
    artifact_file: str = "act.xml",
) -> Path:
    export_dir = tmp_path / "exports_manifest"
    export_dir.mkdir(exist_ok=True)

    runner_spec: dict = (
        {
            "class_path": "physicalai.inference.runners.ActionChunking",
            "init_args": {
                "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
                "chunk_size": chunk_size,
            },
        }
        if kind == "action_chunking"
        else {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}}
    )

    manifest = {
        "format": "policy_package",
        "version": "1.0",
        "policy": {"name": "act", "kind": kind, "class_path": "physicalai.policies.act.ACT"},
        "artifacts": {backend: artifact_file},
        "runner": runner_spec,
    }
    with (export_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f)

    (export_dir / artifact_file).touch()
    if artifact_file.endswith(".xml"):
        (export_dir / artifact_file.replace(".xml", ".bin")).touch()
    return export_dir


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.input_names = ["state", "images"]
    adapter.output_names = ["actions"]
    adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
    adapter.default_device.return_value = "cpu"
    return adapter


@pytest.fixture
def sample_observation() -> dict[str, np.ndarray]:
    return {
        "state": np.random.randn(1, 4).astype(np.float32),
        "images": np.random.randn(1, 3, 224, 224).astype(np.float32),
    }


@pytest.fixture
def _patch_adapter(mock_adapter: MagicMock):
    with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
        yield


@pytest.fixture
def mock_export_dir_no_queue(tmp_path: Path) -> Path:
    return _make_legacy_export_dir(tmp_path, use_action_queue=False, chunk_size=1)


@pytest.mark.usefixtures("_patch_adapter")
class TestInferenceModelInit:
    def test_init_with_valid_directory(self, tmp_path: Path) -> None:
        export_dir = _make_legacy_export_dir(tmp_path)
        model = InferenceModel(export_dir)

        assert model.export_dir == export_dir
        assert model.policy_name == "act"
        assert model.backend == ExportBackend.OPENVINO
        assert model.use_action_queue is True
        assert model.chunk_size == 10
        assert isinstance(model.runner, ActionChunking)

    def test_init_with_nonexistent_directory(self) -> None:
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
        backend_str: str,
        expected: ExportBackend,
        file_ext: str,
    ) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / f"model{file_ext}").touch()
        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": "physicalai.policies.act.ACT"}, f)

        assert InferenceModel(export_dir, backend=backend_str).backend == expected

    def test_load_classmethod(self, tmp_path: Path) -> None:
        model = InferenceModel.load(_make_legacy_export_dir(tmp_path))
        assert isinstance(model, InferenceModel)
        assert model.policy_name == "act"

    def test_init_auto_selects_single_pass_runner(self, tmp_path: Path) -> None:
        export_dir = _make_legacy_export_dir(tmp_path, use_action_queue=False, chunk_size=1)
        assert isinstance(InferenceModel(export_dir).runner, SinglePass)

    def test_init_auto_selects_action_chunking_runner(self, tmp_path: Path) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        assert isinstance(model.runner, ActionChunking)
        assert model.runner.chunk_size == 10

    def test_init_with_explicit_runner(self, tmp_path: Path) -> None:
        explicit_runner = SinglePass()
        model = InferenceModel(_make_legacy_export_dir(tmp_path), runner=explicit_runner)
        assert model.runner is explicit_runner


@pytest.mark.usefixtures("_patch_adapter")
class TestManifestModelInit:
    def test_init_from_manifest_json(self, tmp_path: Path) -> None:
        model = InferenceModel(_make_manifest_export_dir(tmp_path))

        assert model.policy_name == "act"
        assert model.backend == ExportBackend.OPENVINO
        assert model.use_action_queue is True
        assert model.chunk_size == 10
        assert isinstance(model.runner, ActionChunking)
        assert model.runner.chunk_size == 10
        assert isinstance(model.runner.runner, SinglePass)

    def test_init_from_manifest_single_pass(self, tmp_path: Path) -> None:
        model = InferenceModel(
            _make_manifest_export_dir(tmp_path, kind="single_pass", backend="onnx", artifact_file="act.onnx"),
        )
        assert model.policy_name == "act"
        assert model.backend == ExportBackend.ONNX
        assert model.use_action_queue is False
        assert isinstance(model.runner, SinglePass)

    def test_manifest_takes_priority_over_metadata_yaml(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports_dual"
        export_dir.mkdir()

        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": "physicalai.policies.old.Old", "backend": "onnx"}, f)

        manifest = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "new_policy", "kind": "single_pass"},
            "artifacts": {"openvino": "model.xml"},
            "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
        }
        with (export_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f)
        (export_dir / "model.xml").touch()
        (export_dir / "model.bin").touch()

        model = InferenceModel(export_dir)
        assert model.policy_name == "new_policy"
        assert model.backend == ExportBackend.OPENVINO

    def test_select_action_with_manifest(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_manifest_export_dir(tmp_path))
        assert isinstance(model.runner, ActionChunking)

        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
        action1 = model.select_action(sample_observation)
        action2 = model.select_action(sample_observation)

        assert action1.shape == (1, 2)
        assert action2.shape == (1, 2)
        mock_adapter.predict.assert_called_once()

    def test_backend_detected_from_manifest_artifacts(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports_artifacts"
        export_dir.mkdir()
        manifest = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "act"},
            "artifacts": {"onnx": "act.onnx"},
        }
        with (export_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f)
        (export_dir / "act.onnx").touch()

        assert InferenceModel(export_dir).backend == ExportBackend.ONNX


@pytest.mark.usefixtures("_patch_adapter")
class TestMetadataLoading:
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
        format_type: str,
        metadata_content: dict,
    ) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()

        if format_type == "yaml":
            with (export_dir / "metadata.yaml").open("w") as f:
                yaml.dump(metadata_content, f)
            (export_dir / "dummy.xml").touch()
        else:
            with (export_dir / "metadata.json").open("w") as f:
                json.dump(metadata_content, f)
            (export_dir / "act.onnx").touch()

        assert InferenceModel(export_dir).metadata == metadata_content

    def test_no_metadata_fallback(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / "model.onnx").touch()
        assert InferenceModel(export_dir).metadata == {}


class TestAutoDetection:
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
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / backend_file).touch()

        expected_device = "CPU" if backend_type == "openvino" else ("cuda" if cuda_available else "cpu")
        mock_adapter.default_device.return_value = expected_device

        with patch("torch.cuda.is_available", return_value=cuda_available):
            with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
                assert InferenceModel(export_dir, device="auto").device == expected_device

    def test_device_setting(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        (export_dir / "model.onnx").touch()

        with patch("physicalai.inference.model.get_adapter", return_value=TestAdapter(device="xpu")):
            assert InferenceModel(export_dir, device="xpu").device == "xpu"


@pytest.mark.usefixtures("_patch_adapter")
class TestSelectAction:
    def test_select_action_no_queue(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path, use_action_queue=False, chunk_size=1))
        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}

        action = model.select_action(sample_observation)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1, 2)
        mock_adapter.predict.assert_called_once()

    def test_select_action_with_queue(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        assert isinstance(model.runner, ActionChunking)

        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
        action1 = model.select_action(sample_observation)
        assert action1.shape == (1, 2)
        assert len(model.runner._action_queue) == 9

        action2 = model.select_action(sample_observation)
        assert action2.shape == (1, 2)
        assert len(model.runner._action_queue) == 8
        mock_adapter.predict.assert_called_once()

    def test_select_action_queue_refill(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        assert isinstance(model.runner, ActionChunking)

        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 10, 2)}
        for _ in range(10):
            model.select_action(sample_observation)

        assert len(model.runner._action_queue) == 0
        mock_adapter.predict.assert_called_once()

        model.select_action(sample_observation)
        assert len(model.runner._action_queue) == 9
        assert mock_adapter.predict.call_count == 2

    def test_reset_clears_queue(self, tmp_path: Path) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        assert isinstance(model.runner, ActionChunking)
        model.runner._action_queue.extend([np.random.randn(1, 2).astype(np.float32) for _ in range(5)])

        model.reset()
        assert len(model.runner._action_queue) == 0

    def test_call_delegates_to_runner(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path, use_action_queue=False, chunk_size=1))
        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}

        outputs = model(sample_observation)

        assert isinstance(outputs, dict)
        assert outputs["action"].shape == (1, 2)

    def test_call_and_select_action_return_same(
        self,
        tmp_path: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path, use_action_queue=False, chunk_size=1))
        fixed_output = np.random.randn(1, 1, 2)

        mock_adapter.predict.return_value = {"actions": fixed_output.copy()}
        outputs_call = model(sample_observation)

        mock_adapter.predict.return_value = {"actions": fixed_output.copy()}
        action_select = model.select_action(sample_observation)

        np.testing.assert_array_equal(outputs_call["action"], action_select)


@pytest.mark.usefixtures("_patch_adapter")
class TestInputPreparation:
    def test_filters_to_adapter_input_names(self, tmp_path: Path, mock_adapter: MagicMock) -> None:
        mock_adapter.input_names = ["state", "images"]
        model = InferenceModel(_make_legacy_export_dir(tmp_path))

        inputs = {
            "state": np.random.randn(1, 4).astype(np.float32),
            "images": np.random.randn(1, 3, 224, 224).astype(np.float32),
            "action": np.random.randn(1, 2).astype(np.float32),
            "episode_index": np.array([0]),
        }
        result = model._prepare_inputs(inputs)

        assert set(result.keys()) == {"state", "images"}
        np.testing.assert_array_equal(result["state"], inputs["state"])

    def test_passthrough_when_no_input_names(self, tmp_path: Path, mock_adapter: MagicMock) -> None:
        mock_adapter.input_names = []
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        inputs = {"state": np.random.randn(1, 4).astype(np.float32)}

        assert model._prepare_inputs(inputs) is inputs

    def test_nested_payload_flattening(self, tmp_path: Path, mock_adapter: MagicMock) -> None:
        model = InferenceModel(_make_legacy_export_dir(tmp_path))
        mock_adapter.input_names = ["state", "images.top"]

        inputs = {
            "state": np.random.randn(1, 4).astype(np.float32),
            "images": {"top": np.random.randn(1, 3, 224, 224).astype(np.float32)},
            "action": np.random.randn(1, 2).astype(np.float32),
        }
        result = model._prepare_inputs(inputs)
        assert set(result.keys()) == {"state", "images.top"}

@pytest.mark.usefixtures("_patch_adapter")
class TestModelPathResolution:
    @pytest.mark.parametrize(
        ("backend", "file_ext"),
        [
            (ExportBackend.OPENVINO, ".xml"),
            (ExportBackend.ONNX, ".onnx"),
            (ExportBackend.TORCH_EXPORT_IR, ".pt2"),
        ],
    )
    def test_get_model_path_with_policy_name(self, tmp_path: Path, backend: ExportBackend, file_ext: str) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        model_file = export_dir / f"act{file_ext}"
        model_file.touch()

        assert InferenceModel(export_dir, policy_name="act", backend=backend)._get_model_path() == model_file

    def test_get_model_path_without_policy_name(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        model_file = export_dir / "model.onnx"
        model_file.touch()

        assert InferenceModel(export_dir, policy_name=None, backend="onnx")._get_model_path() == model_file

    def test_get_model_path_not_found(self, tmp_path: Path) -> None:
        export_dir = tmp_path / "exports"
        export_dir.mkdir()
        with (export_dir / "metadata.yaml").open("w") as f:
            yaml.dump({"policy_class": "physicalai.policies.act.ACT"}, f)

        with pytest.raises(FileNotFoundError, match="No .* model file found"):
            InferenceModel(export_dir, backend="onnx")._get_model_path()


@pytest.mark.usefixtures("_patch_adapter")
class TestRepr:
    def test_repr(self, tmp_path: Path) -> None:
        repr_str = repr(InferenceModel(_make_legacy_export_dir(tmp_path)))
        assert all(s in repr_str for s in ("InferenceModel", "act", "openvino", "ActionChunking"))


class TestGetRunnerFactory:
    def test_returns_single_pass_by_default(self) -> None:
        assert isinstance(get_runner({}), SinglePass)

    def test_returns_single_pass_when_queue_disabled(self) -> None:
        assert isinstance(get_runner({"use_action_queue": False}), SinglePass)

    def test_returns_action_chunking_when_queue_enabled(self) -> None:
        runner = get_runner({"use_action_queue": True, "chunk_size": 5})
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 5
        assert isinstance(runner.runner, SinglePass)

    def test_returns_action_chunking_default_chunk_size(self) -> None:
        runner = get_runner({"use_action_queue": True})
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 1

    def test_manifest_runner_spec_single_pass(self) -> None:
        metadata = {"runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}}}
        assert isinstance(get_runner(metadata), SinglePass)

    def test_manifest_runner_spec_action_chunking(self) -> None:
        metadata = {
            "runner": {
                "class_path": "physicalai.inference.runners.ActionChunking",
                "init_args": {
                    "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
                    "chunk_size": 8,
                },
            },
        }
        runner = get_runner(metadata)
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 8
        assert isinstance(runner.runner, SinglePass)

    def test_manifest_runner_spec_takes_priority_over_legacy(self) -> None:
        metadata = {
            "use_action_queue": True,
            "chunk_size": 99,
            "runner": {"class_path": "physicalai.inference.runners.SinglePass", "init_args": {}},
        }
        assert isinstance(get_runner(metadata), SinglePass)


class TestSinglePass:
    def test_run_squeezes_temporal_dim(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 1, 4)}
        outputs = SinglePass().run(adapter, {"input": np.zeros(1)})
        assert isinstance(outputs, dict)
        assert outputs["action"].shape == (1, 4)

    def test_run_no_squeeze_when_no_temporal_dim(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 4)}
        outputs = SinglePass().run(adapter, {"input": np.zeros(1)})
        assert isinstance(outputs, dict)
        assert outputs["action"].shape == (1, 4)

    def test_run_returns_full_adapter_output(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {
            "actions": np.random.randn(1, 4),
            "extra_key": np.array([42.0]),
        }
        outputs = SinglePass().run(adapter, {"input": np.zeros(1)})
        assert "action" in outputs
        assert "extra_key" in outputs

    def test_reset_is_noop(self) -> None:
        SinglePass().reset()


class TestActionChunking:
    def test_run_enqueues_and_returns_first(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 5, 2)}

        runner = ActionChunking(runner=SinglePass(), chunk_size=5)
        outputs = runner.run(adapter, {"input": np.zeros(1)})

        assert isinstance(outputs, dict)
        assert outputs["action"].shape == (1, 2)
        assert len(runner._action_queue) == 4

    def test_run_returns_from_queue_without_inference(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 3, 2)}

        runner = ActionChunking(runner=SinglePass(), chunk_size=3)
        runner.run(adapter, {"input": np.zeros(1)})
        runner.run(adapter, {"input": np.zeros(1)})
        adapter.predict.assert_called_once()

    def test_run_refills_when_empty(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 2, 2)}

        runner = ActionChunking(runner=SinglePass(), chunk_size=2)
        runner.run(adapter, {"input": np.zeros(1)})
        runner.run(adapter, {"input": np.zeros(1)})
        assert adapter.predict.call_count == 1

        runner.run(adapter, {"input": np.zeros(1)})
        assert adapter.predict.call_count == 2

    def test_reset_clears_queue(self) -> None:
        adapter = MagicMock()
        adapter.output_names = ["actions"]
        adapter.predict.return_value = {"actions": np.random.randn(1, 5, 2)}

        runner = ActionChunking(runner=SinglePass(), chunk_size=5)
        runner.run(adapter, {"input": np.zeros(1)})
        assert len(runner._action_queue) == 4

        runner.reset()
        assert len(runner._action_queue) == 0

    def test_repr(self) -> None:
        runner = ActionChunking(runner=SinglePass(), chunk_size=10)
        assert "ActionChunking" in repr(runner)
        assert "chunk_size=10" in repr(runner)

    def test_reset_delegates_to_inner_runner(self) -> None:
        inner = MagicMock(spec=InferenceRunner)
        runner = ActionChunking(runner=inner, chunk_size=5)
        runner._action_queue.extend([np.zeros((1, 2)) for _ in range(3)])

        runner.reset()

        assert len(runner._action_queue) == 0
        inner.reset.assert_called_once()


class _ScalePreprocessor(Preprocessor):
    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    @override
    def __call__(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v * self.factor if isinstance(v, np.ndarray) else v for k, v in observation.items()}


class _AddKeyPreprocessor(Preprocessor):
    @override
    def __call__(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {**observation, "preprocessed": np.array([1.0])}


class _ScalePostprocessor(Postprocessor):
    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor

    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs["action"] = outputs["action"] * self.factor
        return outputs


class _ClampPostprocessor(Postprocessor):
    @override
    def __call__(self, outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs["action"] = np.clip(outputs["action"], -1.0, 1.0)
        return outputs


class TestPipelineWiring:
    def test_empty_processors_is_noop(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
        sample_observation: dict[str, np.ndarray],
    ) -> None:
        mock_adapter.predict.return_value = {"actions": np.random.randn(1, 1, 2)}
        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            assert model.preprocessors == []
            assert model.postprocessors == []

            outputs = model(sample_observation)
            assert isinstance(outputs, dict)
            assert isinstance(outputs["action"], np.ndarray)

    def test_preprocessors_run_before_prepare_inputs(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.input_names = ["state", "preprocessed"]
        mock_adapter.predict.return_value = {"actions": np.array([[1.0, 2.0]])}

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            model.preprocessors = [_AddKeyPreprocessor()]

            obs = {"state": np.array([1.0]), "images": np.array([2.0])}
            model(obs)

            call_args = mock_adapter.predict.call_args[0][0]
            assert "preprocessed" in call_args
            assert "images" not in call_args

    def test_preprocessors_execute_in_order(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.input_names = []
        mock_adapter.predict.return_value = {"actions": np.array([[1.0]])}

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            model.preprocessors = [
                _ScalePreprocessor(factor=3.0),
                _ScalePreprocessor(factor=2.0),
            ]

            obs = {"state": np.array([1.0])}
            model(obs)

            call_args = mock_adapter.predict.call_args[0][0]
            np.testing.assert_array_almost_equal(call_args["state"], np.array([6.0]))

    def test_postprocessors_transform_runner_output(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.input_names = []
        mock_adapter.predict.return_value = {"actions": np.array([[10.0, -10.0]])}

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            model.postprocessors = [_ClampPostprocessor()]

            obs = {"state": np.array([1.0])}
            outputs = model(obs)

            np.testing.assert_array_equal(outputs["action"], np.array([[1.0, -1.0]]))

    def test_postprocessors_execute_in_order(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.input_names = []
        mock_adapter.predict.return_value = {"actions": np.array([[10.0]])}

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            model.postprocessors = [
                _ScalePostprocessor(factor=0.5),
                _ClampPostprocessor(),
            ]

            obs = {"state": np.array([1.0])}
            outputs = model(obs)

            # 10.0 * 0.5 = 5.0, then clamp to [-1, 1] → 1.0
            np.testing.assert_array_equal(outputs["action"], np.array([[1.0]]))

    def test_full_pipeline_pre_and_post(
        self,
        mock_export_dir_no_queue: Path,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.input_names = []
        mock_adapter.predict.return_value = {"actions": np.array([[4.0]])}

        with patch("physicalai.inference.model.get_adapter", return_value=mock_adapter):
            model = InferenceModel(mock_export_dir_no_queue)
            model.preprocessors = [_ScalePreprocessor(factor=2.0)]
            model.postprocessors = [_ScalePostprocessor(factor=0.25)]

            obs = {"state": np.array([5.0])}
            outputs = model(obs)

            call_args = mock_adapter.predict.call_args[0][0]
            np.testing.assert_array_almost_equal(call_args["state"], np.array([10.0]))
            np.testing.assert_array_almost_equal(outputs["action"], np.array([[1.0]]))

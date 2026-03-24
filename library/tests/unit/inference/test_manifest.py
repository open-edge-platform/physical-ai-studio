# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from physicalai.inference.manifest import (
    RUNNER_CLASS_PATHS,
    CameraSpec,
    ComponentSpec,
    Manifest,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    _build_runner_spec,
    _policy_name_from_class_path,
)
from physicalai.inference.runners import ActionChunking, SinglePass


class TestTensorSpec:
    def test_from_dict_defaults(self) -> None:
        spec = TensorSpec.from_dict({"shape": [14]})
        assert spec.shape == [14]
        assert spec.dtype == "float32"

    def test_from_dict_explicit_dtype(self) -> None:
        spec = TensorSpec.from_dict({"shape": [3, 480, 640], "dtype": "uint8"})
        assert spec.shape == [3, 480, 640]
        assert spec.dtype == "uint8"


class TestRobotSpec:
    def test_from_dict_minimal(self) -> None:
        spec = RobotSpec.from_dict({"name": "main"})
        assert spec.name == "main"
        assert spec.type == ""
        assert spec.state is None
        assert spec.action is None

    def test_from_dict_full(self) -> None:
        spec = RobotSpec.from_dict({
            "name": "main",
            "type": "Koch v1.1",
            "state": {"shape": [14], "dtype": "float32"},
            "action": {"shape": [14], "dtype": "float32"},
        })
        assert spec.name == "main"
        assert spec.type == "Koch v1.1"
        assert spec.state is not None
        assert spec.state.shape == [14]
        assert spec.action is not None
        assert spec.action.shape == [14]


class TestCameraSpec:
    def test_from_dict_minimal(self) -> None:
        spec = CameraSpec.from_dict({"name": "top"})
        assert spec.name == "top"
        assert spec.shape == []
        assert spec.dtype == "uint8"

    def test_from_dict_full(self) -> None:
        spec = CameraSpec.from_dict({"name": "wrist", "shape": [3, 480, 640], "dtype": "uint8"})
        assert spec.name == "wrist"
        assert spec.shape == [3, 480, 640]


class TestPolicySpec:
    def test_from_dict_defaults(self) -> None:
        spec = PolicySpec.from_dict({})
        assert spec.name == ""
        assert spec.kind == "single_pass"
        assert spec.class_path == ""

    def test_from_dict_full(self) -> None:
        spec = PolicySpec.from_dict({
            "name": "act",
            "kind": "action_chunking",
            "class_path": "physicalai.policies.act.ACT",
        })
        assert spec.name == "act"
        assert spec.kind == "action_chunking"
        assert spec.class_path == "physicalai.policies.act.ACT"


class TestComponentSpec:
    def test_from_dict(self) -> None:
        spec = ComponentSpec.from_dict({
            "class_path": "physicalai.inference.runners.SinglePass",
            "init_args": {},
        })
        assert spec.class_path == "physicalai.inference.runners.SinglePass"
        assert spec.init_args == {}

    def test_from_dict_with_init_args(self) -> None:
        spec = ComponentSpec.from_dict({
            "class_path": "physicalai.inference.runners.ActionChunking",
            "init_args": {"chunk_size": 10},
        })
        assert spec.init_args == {"chunk_size": 10}

    def test_instantiate_single_pass(self) -> None:
        spec = ComponentSpec(
            class_path="physicalai.inference.runners.SinglePass",
            init_args={},
        )
        runner = spec.instantiate()
        assert isinstance(runner, SinglePass)

    def test_instantiate_action_chunking_with_nested_runner(self) -> None:
        spec = ComponentSpec(
            class_path="physicalai.inference.runners.ActionChunking",
            init_args={
                "runner": {
                    "class_path": "physicalai.inference.runners.SinglePass",
                    "init_args": {},
                },
                "chunk_size": 5,
            },
        )
        runner = spec.instantiate()
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 5
        assert isinstance(runner.runner, SinglePass)


class TestManifestFromDict:
    @pytest.fixture
    def full_manifest_data(self) -> dict[str, Any]:
        return {
            "format": "policy_package",
            "version": "1.0",
            "policy": {
                "name": "act",
                "kind": "action_chunking",
                "class_path": "physicalai.policies.act.ACT",
            },
            "artifacts": {"openvino": "act.xml"},
            "runner": {
                "class_path": "physicalai.inference.runners.ActionChunking",
                "init_args": {
                    "runner": {
                        "class_path": "physicalai.inference.runners.SinglePass",
                        "init_args": {},
                    },
                    "chunk_size": 10,
                },
            },
            "robots": [
                {
                    "name": "main",
                    "type": "Koch v1.1",
                    "state": {"shape": [14], "dtype": "float32"},
                    "action": {"shape": [14], "dtype": "float32"},
                },
            ],
            "cameras": [
                {"name": "top", "shape": [3, 480, 640], "dtype": "uint8"},
            ],
        }

    def test_full_manifest(self, full_manifest_data: dict[str, Any]) -> None:
        manifest = Manifest.from_dict(full_manifest_data)

        assert manifest.format == "policy_package"
        assert manifest.version == "1.0"
        assert manifest.policy.name == "act"
        assert manifest.policy.kind == "action_chunking"
        assert manifest.artifacts == {"openvino": "act.xml"}
        assert manifest.runner is not None
        assert manifest.runner.class_path == "physicalai.inference.runners.ActionChunking"
        assert len(manifest.robots) == 1
        assert manifest.robots[0].name == "main"
        assert len(manifest.cameras) == 1
        assert manifest.cameras[0].name == "top"

    def test_minimal_manifest(self) -> None:
        manifest = Manifest.from_dict({})

        assert manifest.format == "policy_package"
        assert manifest.version == "1.0"
        assert manifest.policy.name == ""
        assert manifest.runner is None
        assert manifest.robots == []
        assert manifest.cameras == []

    def test_unknown_keys_go_to_extra(self) -> None:
        manifest = Manifest.from_dict({
            "custom_domain_key": "value",
            "another_key": 42,
        })
        assert manifest.extra == {"custom_domain_key": "value", "another_key": 42}

    def test_runner_instantiation_from_manifest(self, full_manifest_data: dict[str, Any]) -> None:
        manifest = Manifest.from_dict(full_manifest_data)
        assert manifest.runner is not None
        runner = manifest.runner.instantiate()
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 10
        assert isinstance(runner.runner, SinglePass)


class TestManifestFromFile:
    def test_load_from_file_path(self, tmp_path: Path) -> None:
        manifest_data = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "act", "kind": "single_pass"},
            "artifacts": {"onnx": "act.onnx"},
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        manifest = Manifest.load(manifest_path)
        assert manifest.policy.name == "act"
        assert manifest.artifacts == {"onnx": "act.onnx"}

    def test_load_from_directory(self, tmp_path: Path) -> None:
        manifest_data = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "diffusion", "kind": "action_chunking"},
            "artifacts": {"onnx": "diffusion.onnx"},
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        manifest = Manifest.load(tmp_path)
        assert manifest.policy.name == "diffusion"

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            Manifest.load(tmp_path / "nonexistent.json")

    def test_load_directory_without_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            Manifest.load(tmp_path)


class TestManifestFromLegacyMetadata:
    def test_single_pass_policy(self) -> None:
        metadata = {
            "policy_class": "physicalai.policies.act.policy.ACT",
            "backend": "openvino",
            "use_action_queue": False,
            "chunk_size": 1,
        }
        manifest = Manifest.from_legacy_metadata(metadata)

        assert manifest.policy.name == "policy"
        assert manifest.policy.kind == "single_pass"
        assert manifest.policy.class_path == "physicalai.policies.act.policy.ACT"
        assert manifest.runner is not None
        assert "SinglePass" in manifest.runner.class_path

    def test_action_chunking_policy(self) -> None:
        metadata = {
            "policy_class": "physicalai.policies.pi0.policy.Pi0",
            "backend": "onnx",
            "use_action_queue": True,
            "chunk_size": 10,
        }
        manifest = Manifest.from_legacy_metadata(metadata)

        assert manifest.policy.kind == "action_chunking"
        assert manifest.runner is not None
        assert "ActionChunking" in manifest.runner.class_path
        assert manifest.runner.init_args["chunk_size"] == 10

    def test_legacy_extra_preserved(self) -> None:
        metadata = {
            "policy_class": "test.Policy",
            "backend": "openvino",
            "physicalai_train_version": "1.2.3",
        }
        manifest = Manifest.from_legacy_metadata(metadata)
        assert manifest.extra["physicalai_train_version"] == "1.2.3"

    def test_empty_metadata(self) -> None:
        manifest = Manifest.from_legacy_metadata({})
        assert manifest.policy.kind == "single_pass"
        assert manifest.runner is not None


class TestManifestSerialization:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = Manifest(
            policy=PolicySpec(name="act", kind="single_pass", class_path="test.ACT"),
            artifacts={"openvino": "act.xml"},
            runner=ComponentSpec(
                class_path="physicalai.inference.runners.SinglePass",
                init_args={},
            ),
            robots=[RobotSpec(name="main", type="Koch", state=TensorSpec(shape=[14]))],
            cameras=[CameraSpec(name="top", shape=[3, 480, 640])],
        )

        path = tmp_path / "manifest.json"
        original.save(path)

        loaded = Manifest.load(path)
        assert loaded.policy.name == "act"
        assert loaded.artifacts == {"openvino": "act.xml"}
        assert loaded.runner is not None
        assert loaded.runner.class_path == "physicalai.inference.runners.SinglePass"
        assert len(loaded.robots) == 1
        assert loaded.robots[0].name == "main"
        assert len(loaded.cameras) == 1
        assert loaded.cameras[0].name == "top"

    def test_to_dict_omits_empty_optional_sections(self) -> None:
        manifest = Manifest(policy=PolicySpec(name="test"))
        data = manifest.to_dict()

        assert "robots" not in data
        assert "cameras" not in data
        assert "runner" not in data
        assert "adapter" not in data
        assert data["policy"]["name"] == "test"


class TestBuildRunnerSpec:
    def test_single_pass(self) -> None:
        spec = _build_runner_spec("single_pass")
        assert "SinglePass" in spec.class_path
        assert spec.init_args == {}

    def test_action_chunking(self) -> None:
        spec = _build_runner_spec("action_chunking", chunk_size=5)
        assert "ActionChunking" in spec.class_path
        assert spec.init_args["chunk_size"] == 5
        inner = spec.init_args["runner"]
        assert isinstance(inner, dict)
        assert "SinglePass" in inner["class_path"]

    def test_unknown_kind_falls_back_to_single_pass(self) -> None:
        spec = _build_runner_spec("unknown_kind")
        assert "SinglePass" in spec.class_path


class TestPolicyNameFromClassPath:
    @pytest.mark.parametrize(
        ("class_path", "expected"),
        [
            ("physicalai.policies.act.policy.ACT", "policy"),
            ("physicalai.policies.pi0.Pi0", "pi0"),
            ("ab", ""),
            ("", ""),
        ],
    )
    def test_extraction(self, class_path: str, expected: str) -> None:
        assert _policy_name_from_class_path(class_path) == expected


class TestRunnerClassPaths:
    def test_single_pass_path(self) -> None:
        assert RUNNER_CLASS_PATHS["single_pass"] == "physicalai.inference.runners.SinglePass"

    def test_action_chunking_path(self) -> None:
        assert RUNNER_CLASS_PATHS["action_chunking"] == "physicalai.inference.runners.ActionChunking"

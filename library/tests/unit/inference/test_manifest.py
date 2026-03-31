# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from physicalai.inference.component_factory import (
    ComponentRegistry,
    component_registry,
    instantiate_component,
)
from physicalai.inference.manifest import (
    CameraSpec,
    ComponentSpec,
    HardwareSpec,
    Manifest,
    MetadataSpec,
    ModelSpec,
    OrderedTensorSpec,
    PolicySource,
    PolicySpec,
    RobotSpec,
    TensorSpec,
    _policy_name_from_class_path,
)
from physicalai.inference.runners import ActionChunking, SinglePass


class TestTensorSpec:
    def test_from_dict_defaults(self) -> None:
        spec = TensorSpec.model_validate({"shape": [14]})
        assert spec.shape == [14]
        assert spec.dtype == "float32"

    def test_from_dict_explicit_dtype(self) -> None:
        spec = TensorSpec.model_validate({"shape": [3, 480, 640], "dtype": "uint8"})
        assert spec.shape == [3, 480, 640]
        assert spec.dtype == "uint8"


class TestOrderedTensorSpec:
    def test_defaults_to_empty_order(self) -> None:
        spec = OrderedTensorSpec(shape=[14])
        assert spec.order == []
        assert spec.dtype == "float32"

    def test_with_explicit_order(self) -> None:
        joints = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        spec = OrderedTensorSpec(shape=[6], order=joints)
        assert spec.order == joints
        assert spec.shape == [6]

    def test_inherits_shape_validator(self) -> None:
        with pytest.raises(ValidationError):
            OrderedTensorSpec(shape=[-1, 3])

    def test_inherits_dtype_validator(self) -> None:
        with pytest.raises(ValidationError):
            OrderedTensorSpec(shape=[3], dtype="")

    def test_is_subclass_of_tensor_spec(self) -> None:
        spec = OrderedTensorSpec(shape=[6], order=["a", "b", "c", "d", "e", "f"])
        assert isinstance(spec, TensorSpec)

    def test_exclude_defaults_omits_empty_order(self) -> None:
        spec = OrderedTensorSpec(shape=[14])
        data = spec.model_dump(exclude_defaults=True)
        assert "order" not in data
        assert data["shape"] == [14]

    def test_exclude_defaults_keeps_nonempty_order(self) -> None:
        spec = OrderedTensorSpec(shape=[2], order=["j1", "j2"])
        data = spec.model_dump(exclude_defaults=True)
        assert data["order"] == ["j1", "j2"]


class TestRobotSpec:
    def test_from_dict_minimal(self) -> None:
        spec = RobotSpec.model_validate({"name": "main"})
        assert spec.name == "main"
        assert spec.type == ""
        assert spec.state is None
        assert spec.action is None

    def test_from_dict_full(self) -> None:
        spec = RobotSpec.model_validate({
            "name": "main",
            "type": "Koch v1.1",
            "state": {"shape": [14], "dtype": "float32"},
            "action": {"shape": [14], "dtype": "float32"},
        })
        assert spec.name == "main"
        assert spec.type == "Koch v1.1"
        assert spec.state is not None
        assert isinstance(spec.state, OrderedTensorSpec)
        assert spec.state.shape == [14]
        assert spec.state.order == []
        assert spec.action is not None
        assert isinstance(spec.action, OrderedTensorSpec)

    def test_from_dict_with_order(self) -> None:
        joints = ["j1", "j2", "j3"]
        spec = RobotSpec.model_validate({
            "name": "main",
            "state": {"shape": [3], "order": joints},
            "action": {"shape": [3], "order": ["a1", "a2", "a3"]},
        })
        assert spec.state is not None
        assert spec.state.order == joints
        assert spec.action is not None
        assert spec.action.order == ["a1", "a2", "a3"]


class TestCameraSpec:
    def test_from_dict_minimal(self) -> None:
        spec = CameraSpec.model_validate({"name": "top"})
        assert spec.name == "top"
        assert spec.shape == []
        assert spec.dtype == "uint8"

    def test_from_dict_full(self) -> None:
        spec = CameraSpec.model_validate({"name": "wrist", "shape": [3, 480, 640], "dtype": "uint8"})
        assert spec.name == "wrist"
        assert spec.shape == [3, 480, 640]


class TestPolicySource:
    def test_defaults(self) -> None:
        source = PolicySource()
        assert source.repo_id == ""
        assert source.class_path == ""

    def test_from_dict(self) -> None:
        source = PolicySource.model_validate({
            "repo_id": "lerobot/act_aloha",
            "class_path": "physicalai.policies.act.ACT",
        })
        assert source.repo_id == "lerobot/act_aloha"
        assert source.class_path == "physicalai.policies.act.ACT"


class TestPolicySpec:
    def test_from_dict_defaults(self) -> None:
        spec = PolicySpec.model_validate({})
        assert spec.name == ""
        assert spec.source.repo_id == ""
        assert spec.source.class_path == ""

    def test_from_dict_full(self) -> None:
        spec = PolicySpec.model_validate({
            "name": "act",
            "source": {
                "repo_id": "lerobot/act_aloha",
                "class_path": "physicalai.policies.act.ACT",
            },
        })
        assert spec.name == "act"
        assert spec.source.repo_id == "lerobot/act_aloha"
        assert spec.source.class_path == "physicalai.policies.act.ACT"


class TestComponentSpec:
    def test_class_path_mode(self) -> None:
        spec = ComponentSpec.model_validate({
            "class_path": "physicalai.inference.runners.SinglePass",
            "init_args": {},
        })
        assert spec.class_path == "physicalai.inference.runners.SinglePass"
        assert spec.init_args == {}
        assert spec.type == ""

    def test_class_path_with_init_args(self) -> None:
        spec = ComponentSpec.model_validate({
            "class_path": "physicalai.inference.runners.ActionChunking",
            "init_args": {"chunk_size": 10},
        })
        assert spec.init_args == {"chunk_size": 10}

    def test_type_mode(self) -> None:
        spec = ComponentSpec.model_validate({
            "type": "action_chunking",
            "chunk_size": 100,
            "n_action_steps": 100,
        })
        assert spec.type == "action_chunking"
        assert spec.class_path == ""
        assert spec.flat_params == {"chunk_size": 100, "n_action_steps": 100}

    def test_type_mode_no_extra_params(self) -> None:
        spec = ComponentSpec.model_validate({"type": "single_pass"})
        assert spec.type == "single_pass"
        assert spec.flat_params == {}

    def test_requires_type_or_class_path(self) -> None:
        with pytest.raises(ValidationError, match="requires either"):
            ComponentSpec.model_validate({})

    def test_class_path_takes_precedence(self) -> None:
        spec = ComponentSpec.model_validate({
            "type": "single_pass",
            "class_path": "physicalai.inference.runners.SinglePass",
            "init_args": {"foo": "bar"},
        })
        assert spec.class_path == "physicalai.inference.runners.SinglePass"
        assert spec.type == "single_pass"
        assert spec.init_args == {"foo": "bar"}


class TestInstantiateComponent:
    def test_instantiate_single_pass_class_path(self) -> None:
        spec = ComponentSpec(
            class_path="physicalai.inference.runners.SinglePass",
            init_args={},
        )
        runner = instantiate_component(spec)
        assert isinstance(runner, SinglePass)

    def test_instantiate_nested_class_path(self) -> None:
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
        runner = instantiate_component(spec)
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 5
        assert isinstance(runner.runner, SinglePass)

    def test_instantiate_type_mode(self) -> None:
        spec = ComponentSpec.model_validate({"type": "single_pass"})
        runner = instantiate_component(spec)
        assert isinstance(runner, SinglePass)

    def test_instantiate_type_mode_with_params(self) -> None:
        spec = ComponentSpec.model_validate({
            "type": "action_chunking",
            "runner": {"type": "single_pass"},
            "chunk_size": 7,
        })
        runner = instantiate_component(spec)
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 7
        assert isinstance(runner.runner, SinglePass)


class TestModelSpec:
    def test_defaults(self) -> None:
        spec = ModelSpec()
        assert spec.n_obs_steps == 1
        assert spec.runner is None
        assert spec.artifacts == {}
        assert spec.preprocessors == []
        assert spec.postprocessors == []

    def test_from_dict_full(self) -> None:
        spec = ModelSpec.model_validate({
            "n_obs_steps": 2,
            "runner": {"type": "action_chunking", "chunk_size": 100},
            "artifacts": {"model": "model.onnx"},
            "preprocessors": [
                {"class_path": "myapp.transforms.Normalize", "init_args": {"mean": 0.5}},
            ],
            "postprocessors": [
                {"class_path": "myapp.transforms.Clamp", "init_args": {"low": -1.0, "high": 1.0}},
            ],
        })
        assert spec.n_obs_steps == 2
        assert spec.runner is not None
        assert spec.runner.type == "action_chunking"
        assert spec.artifacts == {"model": "model.onnx"}
        assert len(spec.preprocessors) == 1
        assert len(spec.postprocessors) == 1


class TestHardwareSpec:
    def test_defaults(self) -> None:
        spec = HardwareSpec()
        assert spec.robots == []
        assert spec.cameras == []

    def test_from_dict(self) -> None:
        spec = HardwareSpec.model_validate({
            "robots": [{"name": "main", "type": "Koch v1.1"}],
            "cameras": [{"name": "top", "shape": [3, 480, 640]}],
        })
        assert len(spec.robots) == 1
        assert spec.robots[0].name == "main"
        assert len(spec.cameras) == 1
        assert spec.cameras[0].name == "top"


class TestMetadataSpec:
    def test_defaults(self) -> None:
        spec = MetadataSpec()
        assert spec.created_at == ""
        assert spec.created_by == ""

    def test_from_dict(self) -> None:
        spec = MetadataSpec.model_validate({
            "created_at": "2026-01-01T00:00:00Z",
            "created_by": "physicalai",
        })
        assert spec.created_at == "2026-01-01T00:00:00Z"
        assert spec.created_by == "physicalai"


class TestManifestFromDict:
    @pytest.fixture
    def full_manifest_data(self) -> dict[str, Any]:
        return {
            "format": "policy_package",
            "version": "1.0",
            "policy": {
                "name": "act",
                "source": {
                    "repo_id": "lerobot/act_aloha",
                    "class_path": "physicalai.policies.act.ACT",
                },
            },
            "model": {
                "n_obs_steps": 1,
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
                "artifacts": {"openvino": "act.xml"},
            },
            "hardware": {
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
            },
        }

    def test_full_manifest(self, full_manifest_data: dict[str, Any]) -> None:
        manifest = Manifest.model_validate(full_manifest_data)

        assert manifest.format == "policy_package"
        assert manifest.version == "1.0"
        assert manifest.policy.name == "act"
        assert manifest.policy.source.class_path == "physicalai.policies.act.ACT"
        assert manifest.model.artifacts == {"openvino": "act.xml"}
        assert manifest.model.runner is not None
        assert manifest.model.runner.class_path == "physicalai.inference.runners.ActionChunking"
        assert len(manifest.hardware.robots) == 1
        assert manifest.hardware.robots[0].name == "main"
        assert len(manifest.hardware.cameras) == 1
        assert manifest.hardware.cameras[0].name == "top"

    def test_minimal_manifest(self) -> None:
        manifest = Manifest.model_validate({})

        assert manifest.format == "policy_package"
        assert manifest.version == "1.0"
        assert manifest.policy.name == ""
        assert manifest.model.runner is None
        assert manifest.model.artifacts == {}
        assert manifest.model.preprocessors == []
        assert manifest.model.postprocessors == []
        assert manifest.hardware.robots == []
        assert manifest.hardware.cameras == []

    def test_unknown_keys_go_to_extra(self) -> None:
        manifest = Manifest.model_validate({
            "custom_domain_key": "value",
            "another_key": 42,
        })
        assert manifest.model_extra == {"custom_domain_key": "value", "another_key": 42}

    def test_preprocessors_parsed(self) -> None:
        manifest = Manifest.model_validate({
            "model": {
                "preprocessors": [
                    {"class_path": "myapp.transforms.Normalize", "init_args": {"mean": 0.5}},
                    {"class_path": "myapp.transforms.Resize", "init_args": {}},
                ],
            },
        })
        assert len(manifest.model.preprocessors) == 2
        assert manifest.model.preprocessors[0].class_path == "myapp.transforms.Normalize"
        assert manifest.model.preprocessors[0].init_args == {"mean": 0.5}
        assert manifest.model.preprocessors[1].class_path == "myapp.transforms.Resize"

    def test_postprocessors_parsed(self) -> None:
        manifest = Manifest.model_validate({
            "model": {
                "postprocessors": [
                    {"class_path": "myapp.transforms.Clamp", "init_args": {"low": -1.0, "high": 1.0}},
                ],
            },
        })
        assert len(manifest.model.postprocessors) == 1
        assert manifest.model.postprocessors[0].class_path == "myapp.transforms.Clamp"
        assert manifest.model.postprocessors[0].init_args == {"low": -1.0, "high": 1.0}

    def test_runner_instantiation_from_manifest(self, full_manifest_data: dict[str, Any]) -> None:
        manifest = Manifest.model_validate(full_manifest_data)
        assert manifest.model.runner is not None
        runner = instantiate_component(manifest.model.runner)
        assert isinstance(runner, ActionChunking)
        assert runner.chunk_size == 10
        assert isinstance(runner.runner, SinglePass)

    def test_type_based_runner_in_manifest(self) -> None:
        manifest = Manifest.model_validate({
            "model": {
                "runner": {"type": "action_chunking", "chunk_size": 50},
            },
        })
        assert manifest.model.runner is not None
        assert manifest.model.runner.type == "action_chunking"
        assert manifest.model.runner.flat_params == {"chunk_size": 50}


class TestManifestFromFile:
    def test_load_from_file_path(self, tmp_path: Path) -> None:
        manifest_data = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "act"},
            "model": {"artifacts": {"onnx": "act.onnx"}},
        }
        manifest_path = tmp_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        manifest = Manifest.load(manifest_path)
        assert manifest.policy.name == "act"
        assert manifest.model.artifacts == {"onnx": "act.onnx"}

    def test_load_from_directory(self, tmp_path: Path) -> None:
        manifest_data = {
            "format": "policy_package",
            "version": "1.0",
            "policy": {"name": "diffusion"},
            "model": {"artifacts": {"onnx": "diffusion.onnx"}},
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
        assert manifest.policy.source.class_path == "physicalai.policies.act.policy.ACT"
        assert manifest.model.runner is not None
        assert "SinglePass" in manifest.model.runner.class_path

    def test_action_chunking_policy(self) -> None:
        metadata = {
            "policy_class": "physicalai.policies.pi0.policy.Pi0",
            "backend": "onnx",
            "use_action_queue": True,
            "chunk_size": 10,
        }
        manifest = Manifest.from_legacy_metadata(metadata)

        assert manifest.model.runner is not None
        assert "ActionChunking" in manifest.model.runner.class_path
        assert manifest.model.runner.init_args["chunk_size"] == 10

    def test_legacy_extra_preserved(self) -> None:
        metadata = {
            "policy_class": "test.Policy",
            "backend": "openvino",
            "physicalai_train_version": "1.2.3",
        }
        manifest = Manifest.from_legacy_metadata(metadata)
        assert manifest.model_extra is not None
        assert manifest.model_extra["physicalai_train_version"] == "1.2.3"

    def test_empty_metadata(self) -> None:
        manifest = Manifest.from_legacy_metadata({})
        assert manifest.model.runner is not None


class TestManifestSerialization:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = Manifest(
            policy=PolicySpec(
                name="act",
                source=PolicySource(class_path="test.ACT"),
            ),
            model=ModelSpec(
                runner=ComponentSpec(
                    class_path="physicalai.inference.runners.SinglePass",
                    init_args={},
                ),
                artifacts={"openvino": "act.xml"},
            ),
            hardware=HardwareSpec(
                robots=[RobotSpec(name="main", type="Koch", state=OrderedTensorSpec(shape=[14]))],
                cameras=[CameraSpec(name="top", shape=[3, 480, 640])],
            ),
        )

        path = tmp_path / "manifest.json"
        original.save(path)

        loaded = Manifest.load(path)
        assert loaded.policy.name == "act"
        assert loaded.model.artifacts == {"openvino": "act.xml"}
        assert loaded.model.runner is not None
        assert loaded.model.runner.class_path == "physicalai.inference.runners.SinglePass"
        assert len(loaded.hardware.robots) == 1
        assert loaded.hardware.robots[0].name == "main"
        assert len(loaded.hardware.cameras) == 1
        assert loaded.hardware.cameras[0].name == "top"

    def test_to_dict_omits_empty_optional_sections(self) -> None:
        manifest = Manifest(policy=PolicySpec(name="test"))
        data = manifest.model_dump(exclude_defaults=True)

        assert data["policy"]["name"] == "test"

    def test_roundtrip_with_preprocessors_postprocessors(self, tmp_path: Path) -> None:
        original = Manifest(
            policy=PolicySpec(name="act"),
            model=ModelSpec(
                artifacts={"onnx": "act.onnx"},
                preprocessors=[
                    ComponentSpec(class_path="myapp.transforms.Normalize", init_args={"mean": 0.5}),
                ],
                postprocessors=[
                    ComponentSpec(class_path="myapp.transforms.Clamp", init_args={"low": -1.0, "high": 1.0}),
                    ComponentSpec(class_path="myapp.transforms.Scale", init_args={"factor": 2.0}),
                ],
            ),
        )

        path = tmp_path / "manifest.json"
        original.save(path)

        loaded = Manifest.load(path)
        assert len(loaded.model.preprocessors) == 1
        assert loaded.model.preprocessors[0].class_path == "myapp.transforms.Normalize"
        assert loaded.model.preprocessors[0].init_args == {"mean": 0.5}
        assert len(loaded.model.postprocessors) == 2
        assert loaded.model.postprocessors[0].class_path == "myapp.transforms.Clamp"
        assert loaded.model.postprocessors[1].class_path == "myapp.transforms.Scale"
        assert loaded.model.postprocessors[1].init_args == {"factor": 2.0}

    def test_to_dict_includes_nonempty_processors(self) -> None:
        manifest = Manifest(
            model=ModelSpec(
                preprocessors=[ComponentSpec(class_path="a.B", init_args={"x": 1})],
                postprocessors=[ComponentSpec(class_path="c.D", init_args={})],
            ),
        )
        data = manifest.model_dump(exclude_defaults=True)

        model_data = data["model"]
        assert "preprocessors" in model_data
        assert len(model_data["preprocessors"]) == 1
        assert model_data["preprocessors"][0]["class_path"] == "a.B"
        assert model_data["preprocessors"][0]["init_args"] == {"x": 1}

        assert "postprocessors" in model_data
        assert len(model_data["postprocessors"]) == 1
        assert model_data["postprocessors"][0]["class_path"] == "c.D"


class TestComponentSpecFromClass:
    def test_single_pass(self) -> None:
        spec = ComponentSpec.from_class(SinglePass)
        assert "SinglePass" in spec.class_path
        assert spec.init_args == {}

    def test_action_chunking_with_overrides(self) -> None:
        spec = ComponentSpec.from_class(
            ActionChunking,
            runner=ComponentSpec.from_class(SinglePass),
            chunk_size=5,
        )
        assert "ActionChunking" in spec.class_path
        assert spec.init_args["chunk_size"] == 5
        inner = spec.init_args["runner"]
        assert isinstance(inner, dict)
        assert "SinglePass" in inner["class_path"]

    def test_action_chunking_default_chunk_size(self) -> None:
        spec = ComponentSpec.from_class(
            ActionChunking,
            runner=ComponentSpec.from_class(SinglePass),
        )
        assert "ActionChunking" in spec.class_path
        assert spec.init_args["chunk_size"] == 1
        assert isinstance(spec.init_args["runner"], dict)

    def test_missing_required_param_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Missing required parameters"):
            ComponentSpec.from_class(ActionChunking)


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


class TestPydanticValidation:
    def test_tensor_spec_negative_shape(self) -> None:
        with pytest.raises(ValidationError):
            TensorSpec(shape=[-1, 3])

    def test_tensor_spec_empty_dtype(self) -> None:
        with pytest.raises(ValidationError):
            TensorSpec(shape=[3], dtype="")

    def test_component_spec_empty_both_raises(self) -> None:
        with pytest.raises(ValidationError, match="requires either"):
            ComponentSpec.model_validate({})

    def test_component_spec_type_only_is_valid(self) -> None:
        spec = ComponentSpec.model_validate({"type": "single_pass"})
        assert spec.type == "single_pass"

    def test_component_spec_class_path_only_is_valid(self) -> None:
        spec = ComponentSpec(class_path="single_pass")
        assert spec.class_path == "single_pass"

    def test_camera_spec_wrong_shape_length(self) -> None:
        with pytest.raises(ValidationError):
            CameraSpec(name="top", shape=[3, 480])

    def test_camera_spec_empty_shape_is_valid(self) -> None:
        spec = CameraSpec(name="top")
        assert spec.shape == []


class TestComponentRegistry:
    def test_register_and_resolve(self) -> None:
        reg = ComponentRegistry()
        reg.register("my_comp", "mypackage.module.MyClass")
        assert reg.resolve("my_comp") == "mypackage.module.MyClass"

    def test_resolve_passthrough_for_unregistered(self) -> None:
        reg = ComponentRegistry()
        assert reg.resolve("some.full.ClassPath") == "some.full.ClassPath"

    def test_contains(self) -> None:
        reg = ComponentRegistry()
        reg.register("x", "a.B")
        assert "x" in reg
        assert "y" not in reg

    def test_entries_returns_copy(self) -> None:
        reg = ComponentRegistry()
        reg.register("a", "x.Y")
        entries = reg.entries()
        entries["b"] = "z.W"
        assert "b" not in reg

    def test_component_registry_has_runners(self) -> None:
        assert "single_pass" in component_registry
        assert "action_chunking" in component_registry
        assert component_registry.resolve("single_pass") == "physicalai.inference.runners.SinglePass"
        assert component_registry.resolve("action_chunking") == "physicalai.inference.runners.ActionChunking"

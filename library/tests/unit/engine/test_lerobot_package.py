# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime package and native LeRobot loader tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("physicalai_lerobot_policy_plugin", reason="needs the lerobot extra: uv sync --extra lerobot")

from physicalai.engine import Run
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine


def _write_native_checkpoint(model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"type": "act"}), encoding="utf-8")
    (model_dir / "train_config.json").write_text(
        json.dumps({"policy": {"type": "act"}, "steps": 1}),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"native weights")
    _write_processors(model_dir)


def _write_processors(model_dir: Path, **step_overrides: object) -> None:
    """Write processor configs shaped like LeRobot's, with one registered step and its state."""
    state_file = "policy_preprocessor_step_0_normalizer_processor.safetensors"
    (model_dir / state_file).write_bytes(b"stats")
    step: dict[str, object] = {"registry_name": "normalizer_processor", "config": {}, "state_file": state_file}
    step.update(step_overrides)
    step = {key: value for key, value in step.items() if value is not None}
    (model_dir / "policy_preprocessor.json").write_text(json.dumps({"name": "pre", "steps": [step]}), encoding="utf-8")
    (model_dir / "policy_postprocessor.json").write_text(
        json.dumps({"name": "post", "steps": [{"registry_name": "device_processor", "config": {}}]}),
        encoding="utf-8",
    )


class _Processor:
    def __init__(self, *, add_batch: bool = False) -> None:
        self.inputs: dict[str, Any] | None = None
        self.reset_count = 0
        self.add_batch = add_batch

    def __call__(self, values: Any) -> Any:
        self.inputs = values if isinstance(values, dict) else None
        if self.add_batch and isinstance(values, dict):
            return {
                key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim in (1, 3) else value
                for key, value in values.items()
            }
        return values

    def reset(self) -> None:
        self.reset_count += 1


class _NativePolicy:
    def __init__(self, config: object) -> None:
        self.config = config
        self.actions: list[dict[str, Any]] = []
        self.reset_count = 0

    def to(self, device: str) -> _NativePolicy:  # noqa: ARG002
        return self

    def eval(self) -> _NativePolicy:
        return self

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        self.actions.append(batch)
        action_value = float(len(self.actions))
        return torch.tensor([[action_value, 2.0]])

    def reset(self) -> None:
        self.actions.clear()
        self.reset_count += 1


def test_engine_packages_native_checkpoint_for_runtime_and_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import factory
    from lerobot.utils import import_utils
    from physicalai.inference.manifest import Manifest
    from physicalai.inference.model import InferenceModel

    native_model_dir = tmp_path / "checkpoints" / "last" / "pretrained_model"
    _write_native_checkpoint(native_model_dir)
    output_dir = tmp_path / "exports" / "lerobot"
    run = Run(framework="lerobot", output_dir=tmp_path, checkpoint=native_model_dir)

    package = LeRobotEngine.export(run, output_path=output_dir)

    packaged_native_dir = output_dir / "act.lerobot" / "pretrained_model"
    assert package == output_dir
    assert (packaged_native_dir / "model.safetensors").read_bytes() == b"native weights"
    assert (packaged_native_dir / "train_config.json").read_bytes() == (
        native_model_dir / "train_config.json"
    ).read_bytes()

    manifest = Manifest.load(output_dir)
    assert manifest.policy.name == "act"
    assert manifest.model.artifacts == {"lerobot": "act.lerobot"}
    assert (
        manifest.model.callbacks[0].class_path
        == "physicalai_lerobot_policy_plugin.runtime.LeRobotResetCallback"
    )

    native_config = SimpleNamespace(device="cpu", type="act")
    policy = _NativePolicy(native_config)
    preprocessor = _Processor(add_batch=True)
    postprocessor = _Processor()
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: native_config),
    )

    class _PolicyClass:
        @classmethod
        def from_pretrained(cls, model_path: Path, *, config: object) -> _NativePolicy:  # noqa: ARG003
            assert model_path == packaged_native_dir
            return policy

    monkeypatch.setattr(factory, "get_policy_class", lambda policy_type: _PolicyClass)
    monkeypatch.setattr(
        import_utils,
        "register_third_party_plugins",
        lambda: None,
    )
    import lerobot.policies as policies

    monkeypatch.setattr(policies, "make_pre_post_processors", lambda **kwargs: (preprocessor, postprocessor))

    model = InferenceModel(output_dir, device="cpu")
    observation = {
        "state": np.array([0.1, 0.2], dtype=np.float32),
        "images": {"top": np.zeros((3, 8, 8), dtype=np.float32)},
        "task": "push the block",
    }

    np.testing.assert_array_equal(model.select_action(observation), np.array([1.0, 2.0], dtype=np.float32))
    assert preprocessor.inputs is not None
    assert preprocessor.inputs["observation.state"].shape == (2,)
    assert preprocessor.inputs["observation.images.top"].shape == (3, 8, 8)
    native_batch = policy.actions[-1]
    assert set(native_batch) == {"observation.state", "observation.images.top", "task"}
    assert native_batch["observation.state"].shape == (1, 2)
    assert native_batch["observation.images.top"].shape == (1, 3, 8, 8)
    assert native_batch["task"] == ["push the block"]

    np.testing.assert_array_equal(model.select_action(observation), np.array([2.0, 2.0], dtype=np.float32))
    model.reset()
    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    np.testing.assert_array_equal(model.select_action(observation), np.array([1.0, 2.0], dtype=np.float32))


def test_export_rejects_pretrained_model_symlink_escape(tmp_path: Path) -> None:
    outside_checkpoint = tmp_path / "outside" / "pretrained_model"
    _write_native_checkpoint(outside_checkpoint)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pretrained_model").symlink_to(outside_checkpoint, target_is_directory=True)
    package_dir = tmp_path / "package"

    with pytest.raises(ValueError, match="escapes"):
        LeRobotEngine.export(run_dir, output_path=package_dir)

    assert not package_dir.exists()


def test_runtime_loader_rejects_artifact_symlink_escape(tmp_path: Path) -> None:
    from physicalai.inference.manifest import Manifest, ModelSpec, PolicySpec
    from physicalai.inference.model import InferenceModel

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (package_dir / "act.lerobot").symlink_to(outside_dir, target_is_directory=True)
    Manifest(
        policy=PolicySpec(name="act"),
        model=ModelSpec(artifacts={"lerobot": "act.lerobot"}),
    ).save(package_dir / "manifest.json")

    with pytest.raises(ValueError, match="escapes"):
        InferenceModel(package_dir, device="cpu")


def test_export_rejects_unsafe_policy_directory_component(tmp_path: Path) -> None:
    native_model_dir = tmp_path / "checkpoint" / "pretrained_model"
    _write_native_checkpoint(native_model_dir)
    (native_model_dir / "train_config.json").write_text(
        json.dumps({"policy": {"type": "../outside"}}),
        encoding="utf-8",
    )
    package_dir = tmp_path / "package"

    with pytest.raises(ValueError, match="policy type"):
        LeRobotEngine.export(native_model_dir, output_path=package_dir)

    assert not package_dir.exists()


@pytest.mark.parametrize(
    ("step_overrides", "error"),
    [
        ({"registry_name": "smolvla_new_line_processor"}, None),
        ({"registry_name": "action_tokenizer_processor"}, "disallowed processor step"),
        (
            {"registry_name": None, "class": "lerobot.processor.tokenizer_processor.ActionTokenizerProcessorStep"},
            "disallowed processor class",
        ),
        (
            {"registry_name": None, "class": "lerobot.processor.normalize_processor.NormalizerProcessorStep"},
            None,
        ),
        ({"registry_name": "normalizer_processor", "class": "lerobot.processor.X"}, "not both"),
        ({"state_file": "../outside.safetensors"}, "invalid state file"),
        ({"state_file": "a..b.safetensors"}, "invalid state file"),
        ({"state_file": "weights.pickle"}, "invalid state file"),
        ({"state_file": "missing.safetensors"}, "state file is missing"),
    ],
)
def test_processor_configs_are_checked_before_lerobot_loads_them(
    tmp_path: Path,
    step_overrides: dict[str, object],
    error: str | None,
) -> None:
    from physicalai_lerobot_policy_plugin.policy_support import validate_processor_configs

    model_dir = tmp_path / "pretrained_model"
    _write_native_checkpoint(model_dir)
    _write_processors(model_dir, **step_overrides)

    if error is None:
        validate_processor_configs(model_dir)
    else:
        with pytest.raises((ValueError, FileNotFoundError), match=error):
            validate_processor_configs(model_dir)


def test_processor_validation_rejects_symlinks(tmp_path: Path) -> None:
    from physicalai_lerobot_policy_plugin.policy_support import validate_processor_configs

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"stats")
    model_dir = tmp_path / "pretrained_model"
    _write_native_checkpoint(model_dir)
    state_file = model_dir / "policy_preprocessor_step_0_normalizer_processor.safetensors"
    state_file.unlink()
    state_file.symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        validate_processor_configs(model_dir)


def test_export_rejects_an_incomplete_or_unsupported_checkpoint(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete" / "pretrained_model"
    _write_native_checkpoint(incomplete)
    (incomplete / "policy_postprocessor.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing policy_postprocessor.json"):
        LeRobotEngine.export(incomplete, output_path=tmp_path / "export-incomplete")

    unsupported = tmp_path / "unsupported" / "pretrained_model"
    _write_native_checkpoint(unsupported)
    for name in ("config.json", "train_config.json"):
        path = unsupported / name
        path.write_text(path.read_text(encoding="utf-8").replace('"act"', '"pi0_fast"'), encoding="utf-8")
    with pytest.raises(ValueError, match="LeRobot policy 'pi0_fast' is not supported"):
        LeRobotEngine.export(unsupported, output_path=tmp_path / "export-unsupported")
    assert not (tmp_path / "export-incomplete").exists()
    assert not (tmp_path / "export-unsupported").exists()

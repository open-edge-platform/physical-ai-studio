# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Runtime adapter's observation mapping and worker lifecycle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from physicalai_gr00t_policy_plugin import environments, package, runtime
from physicalai_gr00t_policy_plugin.package import IoSpec

SPEC = IoSpec(
    cameras={"image": "front", "wrist_image": "wrist"},
    state={"x": (0, 1), "gripper": (1, 3)},
    action={"x": (0, 1), "gripper": (1, 2)},
)


def test_runtime_observations_become_gr00t_modalities() -> None:
    front = np.full((4, 5, 3), 7, dtype=np.uint8)
    wrist_chw = np.full((3, 4, 5), 0.5, dtype=np.float32)
    observation = runtime.to_gr00t_observation(
        {
            "images": {"front": front},
            "observation.images.wrist": wrist_chw,
            "state": np.array([0.1, 0.2, 0.3], dtype=np.float64),
            "task": ["stack the cups"],
        },
        SPEC,
    )
    assert observation["video"]["image"].shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(observation["video"]["image"][0], front)
    assert observation["video"]["wrist_image"].dtype == np.uint8
    assert observation["video"]["wrist_image"].shape == (1, 4, 5, 3)
    assert int(observation["video"]["wrist_image"].max()) == 128
    np.testing.assert_allclose(observation["state"]["gripper"], [[0.2, 0.3]])
    assert observation["state"]["x"].dtype == np.float32
    assert observation["task"] == "stack the cups"


@pytest.mark.parametrize(
    ("inputs", "error", "match"),
    [
        ({"images": {"front": np.zeros((2, 2, 3), np.uint8)}, "state": np.zeros(3), "task": "t"}, KeyError, "wrist"),
        (
            {"images": {"front": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8)}, "task": "t"},
            KeyError,
            "state",
        ),
        (
            {
                "images": {"front": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8)},
                "state": np.zeros(2),
                "task": "t",
            },
            ValueError,
            "needs 3",
        ),
    ],
)
def test_incomplete_observations_are_rejected(inputs: dict, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        runtime.to_gr00t_observation(inputs, SPEC)


def test_actions_are_flattened_by_their_modality_slices() -> None:
    chunk = runtime.flatten_actions(
        {"gripper": np.full((4, 1), 9.0, dtype=np.float32), "x": np.arange(4, dtype=np.float32)[:, None]},
        SPEC,
    )
    assert chunk.shape == (1, 4, 2)
    np.testing.assert_array_equal(chunk[0, :, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(chunk[0, :, 1], [9, 9, 9, 9])
    with pytest.raises(ValueError, match="expected"):
        runtime.flatten_actions({"x": np.zeros((4, 1)), "gripper": np.zeros((4, 2))}, SPEC)


class _FakeWorker:
    instances: ClassVar[list[_FakeWorker]] = []

    def __init__(self, python: Path, project: Path) -> None:
        self.python = python
        self.project = project
        self.requests: list[dict[str, Any]] = []
        self.closed = False
        self.info = {
            "ok": True,
            "device": "cuda",
            "video_keys": ["image", "wrist_image"],
            "state_keys": ["x", "gripper"],
            "action_keys": ["x", "gripper"],
            "video_horizon": 1,
            "state_horizon": 1,
        }
        _FakeWorker.instances.append(self)

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(message)
        if message["op"] == "hello":
            return {"ok": True, "framework": "gr00t", "package": "gr00t", "version": "0.1.0"}
        if message["op"] == "load":
            return self.info
        if message["op"] == "predict":
            return {"ok": True, "actions": {"x": np.ones((2, 1), np.float32), "gripper": np.zeros((2, 1), np.float32)}}
        return {"ok": True}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / package.ARTIFACT_NAME
    (root / "model").mkdir(parents=True)
    descriptor = {
        "format_version": 1,
        "policy": "gr00t_n1d7",
        "format": "native",
        "model_dir": "model",
        "embodiment_tag": "libero_sim",
        "io": SPEC.to_dict(),
        "environment": environments.provenance(environments.NATIVE),
        "backbone_source": "Qwen/Qwen3-VL-2B-Instruct",
        "backbone_override": True,
    }
    (root / package.DESCRIPTOR_NAME).write_text(json.dumps(descriptor), encoding="utf-8")
    _FakeWorker.instances.clear()
    monkeypatch.setattr(runtime, "_Worker", _FakeWorker)
    monkeypatch.setattr(
        environments,
        "resolve",
        lambda env, **_: (Path(f"/envs/{env.name}/python"), Path(f"/envs/{env.name}/project"), False),
    )
    return root


def test_adapter_loads_through_the_package_environment_and_seeds_each_call(
    artifact: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = runtime.Gr00tRuntimeAdapter(seed=10)
    with caplog.at_level(logging.WARNING):
        adapter.load(artifact)
    assert "development backbone source" in caplog.text
    worker = _FakeWorker.instances[0]
    assert worker.python == Path("/envs/default/python")
    assert worker.project == Path("/envs/default/project")
    load = worker.requests[1]
    assert load["format"] == "native"
    assert load["path"] == str(artifact / "model")
    assert load["embodiment_tag"] == "libero_sim"

    observation = {
        "images": {"front": np.zeros((2, 2, 3), np.uint8), "wrist": np.zeros((2, 2, 3), np.uint8)},
        "state": np.zeros(3, np.float32),
        "task": "t",
    }
    first = adapter.predict(observation)
    adapter.predict(observation)
    adapter.reset()
    adapter.predict(observation)
    seeds = [request["seed"] for request in worker.requests if request["op"] == "predict"]
    assert seeds == [10, 11, 10]
    assert first["action"].shape == (1, 2, 2)
    adapter.close()
    assert worker.closed


def test_adapter_rejects_models_whose_modalities_do_not_match_the_package(artifact: Path) -> None:
    original = _FakeWorker.__init__

    def mismatched(self: _FakeWorker, python: Path, project: Path) -> None:
        original(self, python, project)
        self.info["action_keys"] = ["x", "y"]

    _FakeWorker.__init__ = mismatched  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="action keys"):
            runtime.Gr00tRuntimeAdapter().load(artifact)
    finally:
        _FakeWorker.__init__ = original  # type: ignore[method-assign]
    assert _FakeWorker.instances[-1].closed


def test_reset_callback_forwards_runtime_resets(artifact: Path) -> None:
    adapter = runtime.Gr00tRuntimeAdapter()
    adapter.load(artifact)
    callback = runtime.Gr00tResetCallback()
    callback.on_load(SimpleNamespace(adapter=adapter))  # type: ignore[arg-type]
    callback.on_reset()
    assert _FakeWorker.instances[0].requests[-1] == {"op": "reset"}
    with pytest.raises(TypeError, match="gr00t Runtime adapter"):
        callback.on_load(SimpleNamespace(adapter=object()))  # type: ignore[arg-type]


def test_adapter_requires_load_before_predict() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        runtime.Gr00tRuntimeAdapter().predict({})
    assert runtime.Gr00tRuntimeAdapter().input_names == []
    assert runtime.Gr00tRuntimeAdapter().output_names == ["action"]


def test_openvino_compile_cache_is_kept_outside_the_package(
    artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    descriptor_path = artifact / package.DESCRIPTOR_NAME
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    (artifact / "openvino").mkdir()
    descriptor.update(
        format="openvino",
        model_dir="openvino",
        environment=environments.provenance(environments.OPENVINO),
    )
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
    monkeypatch.setenv("PHYSICALAI_GR00T_CACHE", str(tmp_path / "cache"))

    adapter = runtime.Gr00tRuntimeAdapter()
    adapter.load(artifact)
    load = _FakeWorker.instances[0].requests[1]
    cache_dir = Path(load["cache_dir"])
    assert load["format"] == "openvino"
    assert cache_dir.is_relative_to(tmp_path / "cache" / "openvino-cache")
    assert not cache_dir.is_relative_to(artifact)
    assert adapter.is_openvino
    assert _FakeWorker.instances[0].python == Path("/envs/robotics-ai-suite-openvino/python")

    explicit = runtime.Gr00tRuntimeAdapter(cache_dir=tmp_path / "explicit")
    explicit.load(artifact)
    assert _FakeWorker.instances[1].requests[1]["cache_dir"] == str(tmp_path / "explicit")


def test_packages_built_from_other_pins_are_refused_before_starting_a_worker(artifact: Path) -> None:
    descriptor_path = artifact / package.DESCRIPTOR_NAME
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["environment"]["key"] = "0" * 64
    descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises(ValueError, match="different environment"):
        runtime.Gr00tRuntimeAdapter().load(artifact)
    assert _FakeWorker.instances == []


def test_workers_reporting_another_framework_or_locked_version_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "gr00t"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    runtime._check_worker({"framework": "gr00t", "package": "gr00t", "version": "0.1.0"}, environments.NATIVE, tmp_path)
    with pytest.raises(RuntimeError, match=r"lock pins 0\.1\.0"):
        runtime._check_worker(
            {"framework": "gr00t", "package": "gr00t", "version": "0.2.0"},
            environments.NATIVE,
            tmp_path,
        )
    with pytest.raises(RuntimeError, match="framework"):
        runtime._check_worker({"framework": "lerobot", "package": "lerobot"}, None, tmp_path)


def test_managed_environment_uses_runtime_isolate_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = SimpleNamespace(python=tmp_path / "python", project=tmp_path / "project")
    calls: list[object] = []

    def fake_prepare(environment: object) -> object:
        calls.append(environment)
        return prepared

    monkeypatch.setattr(environments, "isolate_prepare", fake_prepare)
    adapter = runtime.Gr00tRuntimeAdapter()
    descriptor = {"environment": environments.provenance(environments.NATIVE)}

    python, project, managed = adapter._environment(descriptor)

    assert (python, project, managed) == (prepared.python, prepared.project, environments.NATIVE)
    assert calls == [environments.NATIVE.environment]


def test_user_managed_environment_still_requires_its_gr00t_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHYSICALAI_WORKER_PROJECT", raising=False)
    adapter = runtime.Gr00tRuntimeAdapter(interpreter="/opt/gr00t/bin/python")
    descriptor = {"environment": environments.provenance(environments.NATIVE)}

    with pytest.raises(ValueError, match="Isaac-GR00T checkout"):
        adapter._environment(descriptor)

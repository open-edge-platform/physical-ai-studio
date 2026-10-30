from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from runtime_session.controls.model import ModelControl
from runtime_session.model import ModelInputs, model_inputs
from runtime_session.session import RuntimeSession
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.model import Model
from schemas.runtime_session import ExecutionConfig, ModelControlConfig
from tests.workers.fake_model import FakeModelWorker
from workers.base import WorkerStartupError

from .fixtures import make_environment

SPEC = FeatureSpec(("shoulder_pan.pos", "gripper.pos"), {"cam 0": ImageSpec(120, 160, 3)})


def _export(path: Path, *, joints: int = 2, cameras: tuple[str, ...] = ("images.cam 0",)) -> Path:
    features = [{"init_args": {"ftype": "STATE", "shape": [joints], "name": "state"}}]
    features += [{"init_args": {"ftype": "VISUAL", "shape": [3, 120, 160], "name": name}} for name in cameras]
    features.append({"init_args": {"ftype": "LANGUAGE", "shape": [48], "name": "task"}})
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"model": {"input_features": features}}))
    return path


def _model(tmp_path: Path, **export: object) -> Model:
    root = tmp_path / uuid4().hex
    _export(root / "exports" / "torch", **export)  # type: ignore[arg-type]
    return Model(id=uuid4(), name="Picker", path=str(root), policy="act", properties={}, project_id=uuid4())


@pytest.fixture
def fake_model_worker():
    with patch("runtime_session.model.ModelWorker", FakeModelWorker):
        yield


async def _until(condition, within: float = 10.0) -> None:
    deadline = time.monotonic() + within
    while not condition():
        assert time.monotonic() < deadline, "condition never held"
        await asyncio.sleep(0.02)


def test_an_export_reads_its_cameras_from_the_store(tmp_path: Path) -> None:
    assert model_inputs(_export(tmp_path / "a"), SPEC).images == {"images.cam 0": "cam 0"}
    # A single-camera export takes its one camera as plain "images".
    assert model_inputs(_export(tmp_path / "b", cameras=("images",)), SPEC).images == {"images": "cam 0"}


def test_an_export_for_other_joints_or_cameras_does_not_fit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="3 joints"):
        model_inputs(_export(tmp_path / "a", joints=3), SPEC)
    with pytest.raises(ValueError, match="camera 'wrist'"):
        model_inputs(_export(tmp_path / "b", cameras=("images.wrist",)), SPEC)


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_the_model_drives_the_follower(new_session, fake_cameras, fake_model_worker, tmp_path, mode) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path)
    await session.load_model(model, "torch", "cpu")
    assert session.state().model.name == "Picker"

    start = session.store.read(STATE).value.copy()
    config = ModelControlConfig(task="pick", hz=50, execution=ExecutionConfig(mode=mode), grace_s=0)
    await session.set_control(config)
    assert session.state().control == config
    # The oracle commands state + 1 and up, so a follower it drives keeps climbing.
    await _until(lambda: session.store.read(STATE).value[0] > start[0] + 3)

    await session.unload_model()
    assert session.state().model is None
    assert session.state().control is None


async def test_a_model_needs_an_environment_it_fits(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await new_session()
    with pytest.raises(RuntimeSessionCommandError, match="Load an environment"):
        await session.load_model(_model(tmp_path), "torch", "cpu")
    await session.load_environment(make_environment())
    with pytest.raises(ValueError, match="camera 'wrist'"):
        await session.load_model(_model(tmp_path, cameras=("images.wrist",)), "torch", "cpu")
    with pytest.raises(ValueError, match="no openvino export"):
        await session.load_model(_model(tmp_path), "openvino", "cpu")
    with pytest.raises(ValueError, match="Load a model"):
        await session.set_control(ModelControlConfig())
    assert session.state().model is None


async def test_a_model_that_fails_to_load_is_not_loaded(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path)
    (Path(model.path) / "exports" / "torch" / "broken").touch()
    with pytest.raises(WorkerStartupError, match="does not load"):
        await session.load_model(model, "torch", "cpu")
    assert session.state().model is None


async def test_a_dead_model_worker_unloads_the_model_and_holds(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    await session.set_control(ModelControlConfig(hz=50))
    os.kill(session._model.value._worker.pid, signal.SIGKILL)
    await _until(lambda: session.state().model is None and session.state().control is None)
    # The environment is untouched.
    assert session.state().environment is not None


class _FarModel:
    """Commands a pose far from the follower, as a model heading for its rest pose does."""

    inputs = ModelInputs(state_size=2, images={})

    def predict_action_chunk(self, observation: dict) -> np.ndarray:
        return np.full((10, 2), 50.0, dtype=np.float32)

    def reset(self) -> None:
        pass


def test_the_models_actions_are_due_at_the_end_of_the_grace_period() -> None:
    store = Store.create(FeatureSpec(("shoulder_pan.pos", "gripper.pos")), uuid4().hex)
    try:
        store.write(STATE, [0.0, 10.0], timestamp=time.monotonic())
        control = ModelControl(store, _FarModel(), ModelControlConfig(grace_s=1.0))  # type: ignore[arg-type]
        control.on_start()
        control.on_tick()
        # The model's action, due once the grace period is over: the robot worker closes the gap over it.
        action = store.read(ACTION)
        np.testing.assert_allclose(action.value, [50.0, 50.0])
        assert action.timestamp == pytest.approx(control.started_at + 1.0)

        control.started_at = time.monotonic() - 2.0  # past it: due one tick from now
        control.on_tick()
        assert store.read(ACTION).timestamp == pytest.approx(time.monotonic() + 1 / control.hz, abs=0.01)
        control.on_stop()
    finally:
        store.unlink()

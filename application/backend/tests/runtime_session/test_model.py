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
from tests.workers.fake_model import CHUNK, FakeModelWorker
from workers.base import WorkerStartupError

from .fixtures import make_environment

SPEC = FeatureSpec(("shoulder_pan.pos", "gripper.pos"), {"cam 0": ImageSpec(120, 160, 3)})


def _export(
    path: Path,
    *,
    joints: int = 2,
    cameras: tuple[str, ...] = ("images.cam 0",),
    tasks: list[str] | None = None,
    rtc_chunk: int | None = None,
) -> Path:
    features = [{"init_args": {"ftype": "STATE", "shape": [joints], "name": "state"}}]
    features += [{"init_args": {"ftype": "VISUAL", "shape": [3, 120, 160], "name": name}} for name in cameras]
    features.append({"init_args": {"ftype": "LANGUAGE", "shape": [48], "name": "task"}})
    if rtc_chunk is not None:  # as a pi05 export made with rtc_enabled lists them
        features.append(
            {"init_args": {"ftype": "COMMON", "shape": [rtc_chunk, joints], "name": "prev_chunk_left_over"}}
        )
        features += [
            {"init_args": {"ftype": "COMMON", "shape": [], "name": name}}
            for name in ("inference_delay", "max_guidance_weight", "execution_horizon")
        ]
    path.mkdir(parents=True)
    manifest: dict = {"model": {"input_features": features, "preprocessors": [{"type": "to_float_tensor"}]}}
    if tasks is not None:
        manifest["model"]["preprocessors"].append({"type": "task_index", "tasks": tasks})
    (path / "manifest.json").write_text(json.dumps(manifest))
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


def test_an_export_trained_on_a_fixed_set_of_tasks_lists_them(tmp_path: Path) -> None:
    assert model_inputs(_export(tmp_path / "a"), SPEC).tasks == ()
    assert model_inputs(_export(tmp_path / "b", tasks=["pick", "place"]), SPEC).tasks == ("pick", "place")


def test_a_multi_task_model_only_runs_one_of_its_tasks() -> None:
    store = Store.create(FeatureSpec(("shoulder_pan.pos", "gripper.pos")), uuid4().hex)
    try:
        model = _FarModel()
        model.inputs = ModelInputs(state_size=2, images={}, tasks=("pick", "place"))
        with pytest.raises(ValueError, match="pick, place"):
            ModelControl(store, model, ModelControlConfig(task="dance"))  # type: ignore[arg-type]
        ModelControl(store, model, ModelControlConfig(task="place"))  # type: ignore[arg-type]
    finally:
        store.unlink()


def test_an_export_for_other_joints_or_cameras_does_not_fit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="3 joints"):
        model_inputs(_export(tmp_path / "a", joints=3), SPEC)
    with pytest.raises(ValueError, match="camera 'wrist'"):
        model_inputs(_export(tmp_path / "b", cameras=("images.wrist",)), SPEC)


def test_an_export_with_rtc_inputs_says_its_chunk(tmp_path: Path) -> None:
    assert model_inputs(_export(tmp_path / "a", rtc_chunk=50), SPEC).rtc_chunk == (50, 2)
    assert model_inputs(_export(tmp_path / "b"), SPEC).rtc_chunk is None


@pytest.mark.parametrize(
    ("mode", "rtc_export"), [("sync", False), ("async", False), ("sync", True), ("async", True), ("rtc", True)]
)
async def test_the_model_drives_the_follower(
    new_session, fake_cameras, fake_model_worker, tmp_path, mode, rtc_export
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    # The oracle's chunk is 10 actions; an RTC export of it guides at most 5.
    model = _model(tmp_path, rtc_chunk=CHUNK if rtc_export else None)
    await session.load_model(model, "torch", "cpu")
    assert session.state().model.name == "Picker"
    assert session.state().model.rtc is rtc_export

    start = session.store.read(STATE).value.copy()
    execution = ExecutionConfig(mode=mode, execution_horizon=5)
    config = ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0)
    await session.set_control(config)
    assert session.state().control == config
    # The oracle commands state + 1 and up, so a follower it drives keeps climbing.
    await _until(lambda: session.store.read(STATE).value[0] > start[0] + 3)

    await session.unload_model()
    assert session.state().model is None
    assert session.state().control is None


async def test_switching_modes_keeps_driving_the_follower(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path, rtc_chunk=CHUNK), "torch", "cpu")
    for mode in ("rtc", "sync", "async", "rtc"):
        start = session.store.read(STATE).value.copy()
        execution = ExecutionConfig(mode=mode, execution_horizon=5)
        await session.set_control(ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0))
        await _until(lambda start=start: session.store.read(STATE).value[0] > start[0] + 3)
    await session.set_control(None)
    assert session.state().control is None


async def test_stopping_rtc_while_it_infers_holds_the_follower(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path, rtc_chunk=CHUNK)
    (Path(model.path) / "exports" / "torch" / "slow").touch()
    await session.load_model(model, "torch", "cpu")
    execution = ExecutionConfig(mode="rtc", execution_horizon=5)
    await session.set_control(ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0))
    start = session.store.read(STATE).value.copy()
    await _until(lambda: session.store.read(STATE).value[0] > start[0] + 1)
    started = time.monotonic()
    await session.set_control(None)  # an inference is in flight most of the time
    assert time.monotonic() - started < 10
    assert session.state().control is None
    held = session.store.read(ACTION).value.copy()
    await asyncio.sleep(1.0)
    np.testing.assert_allclose(session.store.read(ACTION).value, held)
    # The model is still loaded and runs again.
    await session.set_control(ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0))
    again = session.store.read(STATE).value.copy()
    await _until(lambda: session.store.read(STATE).value[0] > again[0] + 1)


async def test_rtc_refuses_an_export_without_rtc_inputs(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    with pytest.raises(ValueError, match="without RTC inputs"):
        await session.set_control(ModelControlConfig(execution=ExecutionConfig(mode="rtc")))
    assert session.state().control is None


async def test_rtc_refuses_a_horizon_past_half_the_chunk(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path, rtc_chunk=CHUNK), "torch", "cpu")
    with pytest.raises(ValueError, match="at most half the chunk"):
        await session.set_control(ModelControlConfig(execution=ExecutionConfig(mode="rtc", execution_horizon=6)))


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

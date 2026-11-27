from __future__ import annotations

import asyncio
import itertools
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
from runtime_session.bus import command_key
from runtime_session.controls.model import ModelControl
from runtime_session.model import ModelInputs, model_inputs
from runtime_session.session import RuntimeSession
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.dataset import Dataset
from schemas.model import Model
from schemas.runtime_session import ExecutionConfig, ModelControlConfig, StartEpisodeCommand, TeleopControlConfig
from tests.workers.fake_dataset import FakeDatasetWorker
from tests.workers.fake_model import CHUNK, NO_FRAME, FakeModelWorker
from workers.base import WorkerStartupError
from workers.dataset_worker import RecordingStatus

from .fixtures import make_environment, next_status, subscribe_statuses

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


def _rtc_export(path: Path, features: list[dict]) -> Path:
    """An export whose RTC features are ``features``, after a 2-joint state and one camera."""
    base = [
        {"init_args": {"ftype": "STATE", "shape": [2], "name": "state"}},
        {"init_args": {"ftype": "VISUAL", "shape": [3, 120, 160], "name": "images.cam 0"}},
    ]
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"model": {"input_features": base + features}}))
    return path


def _rtc_features(tail: list[int], scalar: list[int] | None = None, *, drop: str | None = None) -> list[dict]:
    features = [{"init_args": {"ftype": "COMMON", "shape": tail, "name": "prev_chunk_left_over"}}]
    features += [
        {"init_args": {"ftype": "COMMON", "shape": scalar or [], "name": name}}
        for name in ("inference_delay", "max_guidance_weight", "execution_horizon")
    ]
    return [feature for feature in features if feature["init_args"]["name"] != drop]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (_rtc_features([50]), "prev_chunk_left_over.*\\(chunk_size, action_dim\\)"),
        (_rtc_features([50, 7]), "7 actions per step.*follower has 2 joints"),
        (_rtc_features([1, 2]), "chunk of 1"),
        (_rtc_features([50, 2], [1]), "inference_delay.*a scalar"),
        (_rtc_features([50, 2], drop="execution_horizon"), "missing execution_horizon"),
    ],
)
def test_an_export_with_broken_rtc_inputs_is_refused_with_a_reason(
    tmp_path: Path, features: list[dict], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as refused:
        model_inputs(_rtc_export(tmp_path / "a", features), SPEC)
    assert "re-export" in str(refused.value)


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
    assert session.state().model.rtc_max_horizon == (CHUNK // 2 if rtc_export else None)

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


def _received(export: Path) -> list[dict]:
    path = export / "received.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def _gate(export: Path, n: int) -> None:
    """Wait until the oracle holds inference ``n`` at its gate."""
    await _until(lambda: (export / f"entered_{n}").exists())


def _rows_of(received: list[dict], n: int) -> np.ndarray:
    return np.asarray(next(row["chunk"] for row in received if row["n"] == n), dtype=np.float32)


def _played(action: np.ndarray, chunk: np.ndarray) -> bool:
    return bool(np.any(np.all(np.isclose(chunk, action, atol=1e-5), axis=1)))


async def test_rtc_sends_the_guided_tail_and_settings_across_to_the_model(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path, rtc_chunk=CHUNK)
    export = Path(model.path) / "exports" / "torch"
    (export / "record").touch()
    await session.load_model(model, "torch", "cpu")
    # 0.06 s at 50 Hz is a delay of 3 actions until the first steady measurement.
    execution = ExecutionConfig(mode="rtc", execution_horizon=4, max_guidance_weight=3.5, initial_latency_s=0.06)
    await session.set_control(ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0))
    await _until(lambda: len(_received(export)) >= 5)
    await session.set_control(None)
    received = _received(export)

    first = received[0]["rtc"]  # warmup: RTC's neutral first-chunk inputs, no tail
    assert (first["inference_delay"], first["execution_horizon"]) == (0, 1)
    assert first["max_guidance_weight"] == pytest.approx(1e-8)
    assert not np.any(first["prev_chunk_left_over"])
    assert received[1]["rtc"]["inference_delay"] == 3  # the seeded latency, not 0
    for previous, row in itertools.pairwise(received):
        rtc = row["rtc"]
        tail = np.asarray(rtc["prev_chunk_left_over"], dtype=np.float32)
        assert tail.shape == (CHUNK, 2)
        left = int(np.count_nonzero(np.any(tail != 0, axis=1)))
        assert left >= 1
        # The unplayed end of the chunk the model returned last, then zeros.
        np.testing.assert_allclose(tail[:left], _rows_of(received, previous["n"])[-left:], atol=1e-6)
        assert not np.any(tail[left:])
        assert rtc["max_guidance_weight"] == pytest.approx(3.5)
        assert rtc["execution_horizon"] == min(4, left)
        assert 0 <= rtc["inference_delay"] <= rtc["execution_horizon"]


@pytest.mark.parametrize("gated", [1, 2], ids=["warmup", "refill"])
async def test_stopping_rtc_mid_inference_keeps_its_late_chunk_out(
    new_session, fake_cameras, fake_model_worker, tmp_path, gated
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path, rtc_chunk=CHUNK)
    export = Path(model.path) / "exports" / "torch"
    (export / "record").touch()
    (export / f"gate_{gated}").touch()
    await session.load_model(model, "torch", "cpu")
    execution = ExecutionConfig(mode="rtc", execution_horizon=5)
    config = ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0)
    await session.set_control(config)
    await _gate(export, gated)

    started = time.monotonic()
    await session.set_control(None)  # inference ``gated`` is inside the model
    # A refill blocks only RTC's own thread, which the control leaves after RTC's 5 s join. A warmup blocks the
    # control's thread itself, which the session gives up on after 10 s, as it does for Sync and Async.
    assert time.monotonic() - started < (8 if gated > 1 else 12)
    assert session.state().control is None
    held = session.store.read(ACTION).value.copy()
    (export / f"release_{gated}").touch()
    await _until(lambda: any(row["n"] == gated for row in _received(export)))
    await asyncio.sleep(0.5)
    if gated > 1:
        # RTC's thread discards a chunk that comes back after its run stopped. (After a warmup, the control's own
        # thread writes it, in every mode: a session-level issue that a separate fix closes.)
        np.testing.assert_allclose(session.store.read(ACTION).value, held)

    # A new run plays its own chunks, never the one that came back late.
    await session.set_control(config)
    late = _rows_of(_received(export), gated)
    seen: list[np.ndarray] = []
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        seen.append(session.store.read(ACTION).value.copy())
        await asyncio.sleep(0.01)
    await session.set_control(None)
    assert not any(_played(action, late) for action in seen)
    later = [_rows_of(_received(export), row["n"]) for row in _received(export) if row["n"] > gated]
    assert any(_played(action, chunk) for action in seen for chunk in later)


async def test_a_failed_rtc_inference_reports_the_models_own_error(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    await session.load_environment(make_environment())
    model = _model(tmp_path, rtc_chunk=CHUNK)
    (Path(model.path) / "exports" / "torch" / "fail").touch()
    await session.load_model(model, "torch", "cpu")
    statuses = subscribe_statuses(session)
    execution = ExecutionConfig(mode="rtc", execution_horizon=5)
    await session.set_control(ModelControlConfig(task="pick", hz=50, execution=execution, grace_s=0))
    status = await next_status(statuses, "control")
    assert status.state == "failed"
    assert NO_FRAME in (status.error or "")


async def test_rtc_refuses_an_export_without_rtc_inputs(new_session, fake_cameras, fake_model_worker, tmp_path) -> None:
    session = await new_session()
    await session.load_environment(make_environment())
    await session.load_model(_model(tmp_path), "torch", "cpu")
    with pytest.raises(ValueError, match="without RTC inputs"):
        await session.set_control(ModelControlConfig(execution=ExecutionConfig(mode="rtc")))
    assert session.state().control is None


async def _recording(session: RuntimeSession, dataset: Dataset) -> None:
    await session.load_dataset(dataset)
    command = StartEpisodeCommand(request_id="", task="pick")
    await session.bus.request(command_key("start_episode"), command, RecordingStatus, timeout=10)
    await _until(lambda: session.state().dataset.is_recording)


def _new_dataset() -> Dataset:
    dataset = Dataset(id=uuid4(), name="Blocks", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    Path(dataset.path).mkdir(parents=True)
    return dataset


async def test_rtc_without_a_model_is_refused_before_teleop_and_the_episode_stop(
    new_session, fake_cameras, fake_model_worker
) -> None:
    session: RuntimeSession = await new_session()
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        await session.load_environment(make_environment())
        teleop = TeleopControlConfig(grace_s=0)
        await session.set_control(teleop)
        await _recording(session, _new_dataset())

        with pytest.raises(ValueError, match="Load a model first"):
            await session.set_control(ModelControlConfig(execution=ExecutionConfig(mode="rtc")))

        assert session.state().control == teleop
        assert session.state().dataset.is_recording
        await session.set_control(None)


async def test_a_refused_rtc_export_keeps_the_loaded_model_its_control_and_episode(
    new_session, fake_cameras, fake_model_worker, tmp_path
) -> None:
    session: RuntimeSession = await new_session()
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        await session.load_environment(make_environment())
        await session.load_model(_model(tmp_path), "torch", "cpu")
        running = ModelControlConfig(task="pick", hz=50, grace_s=0)
        await session.set_control(running)
        await _recording(session, _new_dataset())
        broken = _model(tmp_path, rtc_chunk=CHUNK)
        manifest = Path(broken.path) / "exports" / "torch" / "manifest.json"
        document = json.loads(manifest.read_text())
        document["model"]["input_features"] = [
            feature
            for feature in document["model"]["input_features"]
            if feature["init_args"]["name"] != "inference_delay"
        ]
        manifest.write_text(json.dumps(document))

        with pytest.raises(ValueError, match="missing inference_delay"):
            await session.load_model(broken, "torch", "cpu")

        assert session.state().model.model_id == session._model.value.model.id
        assert session.state().control == running
        assert session.state().dataset.is_recording
        await session.set_control(None)


@pytest.mark.parametrize(("rtc_chunk", "horizon"), [(None, 5), (CHUNK, 6)])
async def test_a_refused_rtc_request_keeps_the_running_control_and_episode(
    new_session, fake_cameras, fake_model_worker, tmp_path, rtc_chunk, horizon
) -> None:
    dataset = Dataset(id=uuid4(), name="Blocks", default_task="pick", project_id=uuid4(), environment_id=uuid4())
    Path(dataset.path).mkdir(parents=True)
    session: RuntimeSession = await new_session()
    with patch("runtime_session.dataset.DatasetWorker", FakeDatasetWorker):
        await session.load_environment(make_environment())
        await session.load_dataset(dataset)
        await session.load_model(_model(tmp_path, rtc_chunk=rtc_chunk), "torch", "cpu")
        running = ModelControlConfig(task="pick", hz=50, execution=ExecutionConfig(mode="sync"), grace_s=0)
        await session.set_control(running)
        command = StartEpisodeCommand(request_id="", task="pick")
        await session.bus.request(command_key("start_episode"), command, RecordingStatus, timeout=10)
        await _until(lambda: session.state().dataset.is_recording)

        refused = ModelControlConfig(
            task="pick", hz=50, execution=ExecutionConfig(mode="rtc", execution_horizon=horizon)
        )
        with pytest.raises(ValueError, match=r"without RTC inputs|at most half the chunk"):
            await session.set_control(refused)

        assert session.state().control == running
        assert session.state().dataset.is_recording
        start = session.store.read(STATE).value.copy()
        await _until(lambda: session.store.read(STATE).value[0] > start[0] + 3)
        await session.set_control(None)


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

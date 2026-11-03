from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import rerun as rr
from physicalai.runtime.events import InferenceEvent, MetricsEvent

from runtime_session.controls.model import ModelControl
from runtime_session.rerun import RerunObserver
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import ExecutionConfig, ModelControlConfig, SessionState, StateMessage

from .test_model import _FarModel


@pytest.fixture
def store():
    store = Store.create(FeatureSpec(("shoulder", "gripper"), {"wrist": ImageSpec(4, 4)}), uuid4().hex)
    yield store
    store.unlink()


def _paths(recording: MagicMock) -> list[str]:
    return [call.args[0] for call in recording.log.call_args_list]


async def test_spawn_uses_explicit_port_and_a_recording_per_session() -> None:
    with patch("rerun.spawn") as spawn:
        async with RerunObserver("first") as first, RerunObserver("second") as second:
            assert first._recording is not second._recording
            assert rr.get_recording_id(first._recording) == "first"
            assert rr.get_recording_id(second._recording) == "second"
            assert rr.get_global_data_recording() is None
        assert [call.kwargs["port"] for call in spawn.call_args_list] == [9876, 9876]


def test_samples_targets_and_latest_images_only_when_changed(store: Store) -> None:
    observer = RerunObserver("sample")
    recording = observer._recording = MagicMock()
    now = time.monotonic()
    store.write(STATE, [1, 2], timestamp=now)
    store.write(ACTION, [3, 4], timestamp=now + 1)
    store.write_observation("wrist", np.zeros((4, 4, 3), dtype=np.uint8), timestamp=now)
    observer.set_store(store)
    observer._sample()
    assert "robot/joints" in _paths(recording)
    assert "robot/targets" in _paths(recording)
    assert "camera/wrist" in _paths(recording)
    recording.reset_mock()
    observer._sample()
    recording.log.assert_not_called()

    # Images are capped independently of joints, even if a camera updates faster.
    store.write(STATE, [2, 3], timestamp=now + 0.01)
    store.write_observation("wrist", np.ones((4, 4, 3), dtype=np.uint8), timestamp=now + 0.01)
    observer._sample()
    assert _paths(recording) == ["robot/joints"]
    observer._next_images = 0
    observer._sample()
    assert _paths(recording)[-1] == "camera/wrist"


def test_detaching_releases_store_before_unlink_and_reconfigures_a_replacement(store: Store) -> None:
    observer = RerunObserver("replace")
    observer._recording = MagicMock()
    observer.set_store(store)
    observer._sample()
    observer.set_store(None)
    observer._sample()
    replacement = Store.create(FeatureSpec(("other",)), uuid4().hex)
    try:
        replacement.write(STATE, [42], timestamp=time.monotonic())
        observer.set_store(replacement)
        observer._sample()
        assert observer._last_store is replacement
        assert observer._recording.send_blueprint.call_count == 2
        observer.set_store(None)
    finally:
        replacement.unlink()
    # The observer never touches a detached mapping again.
    observer._sample()


def test_prediction_offset_rate_and_grace_period(store: Store) -> None:
    observer = RerunObserver("infer")
    recording = observer._recording = MagicMock()
    events = observer.control_events(hz=20, grace_s=5)
    events.emit_inference(
        InferenceEvent("infer", time.time(), latency_s=0.1, offset=1, chunk=np.array([[1, 2], [3, 4], [5, 6]]))
    )
    events.emit_metrics(MetricsEvent("infer", 0, time.time(), {"queue_remaining": 2}))
    with patch("rerun.TimeColumn", wraps=rr.TimeColumn) as column:
        observer._drain_events()
    np.testing.assert_allclose(
        column.call_args.kwargs["duration"], events.ready_at - observer._origin + np.arange(2) / 20
    )
    assert "inference/latency_ms" in _paths(recording)
    assert "queue/remaining" in _paths(recording)
    assert list(observer._latencies) == [100]


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_model_control_emits_predictions_and_queue_depth(store: Store, mode: str) -> None:
    observer = RerunObserver("model")
    observer._recording = MagicMock()
    store.write(STATE, [1, 2], timestamp=time.monotonic())
    config = ModelControlConfig(hz=100, grace_s=0, execution=ExecutionConfig(mode=mode))
    async with ModelControl(store, _FarModel(), config, observer=observer):  # type: ignore[arg-type]
        async with asyncio.timeout(5):
            while not any(isinstance(item[0], InferenceEvent) for item in observer._events):  # noqa: ASYNC110
                await asyncio.sleep(0.02)
    observer._drain_events()
    assert "inference/latency_ms" in _paths(observer._recording)
    assert "queue/remaining" in _paths(observer._recording)


def test_event_buffer_is_bounded() -> None:
    observer = RerunObserver("bounded")
    events = observer.control_events(30, 0)
    for step in range(1000):
        events.emit_metrics(MetricsEvent("bounded", step, time.time(), {"queue_remaining": step}))
    assert len(observer._events) == 64
    assert observer._events[-1][0].step == 999


def test_stopping_model_control_clears_live_predictions_and_queue() -> None:
    observer = RerunObserver("hold")
    observer._recording = MagicMock()
    observer.emit_session(StateMessage(data=SessionState(session_id="hold")))
    observer._drain_events()
    assert _paths(observer._recording) == ["robot/predicted", "queue", "runtime/lifecycle"]


def test_system_metrics_are_throttled_and_identify_the_backend_process() -> None:
    observer = RerunObserver("system")
    observer._recording = MagicMock()
    with patch("runtime_session.rerun.psutil.Process") as process:
        process.return_value.cpu_percent.return_value = 200
        process.return_value.memory_info.return_value.rss = 10 * 2**20
        with patch("runtime_session.rerun.psutil.cpu_count", return_value=4):
            observer._sample_system()  # prime counters
            observer._sample_system()  # not due
            observer._recording.log.assert_not_called()
            observer._next_system = 0
            observer._sample_system()
    values = {call.args[0]: call.args[1] for call in observer._recording.log.call_args_list}
    assert "runtime/host/cpu_percent" in values
    assert values["runtime/studio/cpu_percent"] == rr.Scalars(50)
    assert values["runtime/studio/rss_mib"] == rr.Scalars(10)


async def test_real_sdk_accepts_blueprint_images_and_prediction_columns(store: Store) -> None:
    observer = RerunObserver("sdk")
    async with observer.lifecycle():
        stream = observer._recording.binary_stream()
        store.write(STATE, [1, 2], timestamp=time.monotonic())
        store.write_observation("wrist", np.zeros((4, 4, 3), dtype=np.uint8), timestamp=time.monotonic())
        observer.set_store(store)
        rr.set_strict_mode(True)
        try:
            observer._sample()
            observer.control_events(30, 0).emit_inference(InferenceEvent("sdk", time.time(), 0.01, 0, np.zeros((3, 2))))
            observer._drain_events()
        finally:
            rr.set_strict_mode(False)
        assert stream.read(flush=True)


async def test_viewer_start_failure_does_not_prevent_session_use(new_session) -> None:
    with patch("rerun.spawn", side_effect=RuntimeError("no display")):
        session = await new_session()
    assert session._observer is None
    assert session.state().session_id == session.id
    await session.stop()
    assert session.finished.is_set()


async def test_session_detaches_store_and_joins_observer_on_close(new_session, fake_cameras) -> None:
    from .fixtures import make_environment

    session = await new_session()
    observer = session._observer
    await session.load_environment(make_environment())
    assert observer._store is session.store
    await session.stop()
    assert observer._store is None
    assert not observer.is_alive()


async def test_logging_failure_does_not_close_the_session(new_session) -> None:
    session = await new_session()
    observer = session._observer
    with patch.object(observer, "_sample", side_effect=RuntimeError("viewer failed")):
        async with asyncio.timeout(5):
            await observer.wait_stopped()
    assert observer.failure is not None
    assert not session.closed.is_set()
    await session.stop()

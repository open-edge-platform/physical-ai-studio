from __future__ import annotations

import asyncio
import socket
import time
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
import numpy as np
import pytest
import rerun as rr
from physicalai.runtime import RerunCallback
from physicalai.runtime.callbacks.gpu_stats import GpuDeviceStats, GpuStatsSample
from physicalai.runtime.callbacks.system_monitor import SystemStats, SystemStatsSample
from physicalai.runtime.events import InferenceEvent, MetricsEvent, TickEvent

from runtime_session.controls.model import ModelControl, _Observation
from runtime_session.rerun import _BACKGROUND_SERVERS, RerunObserver, _listening, _Prediction
from runtime_session.rerun import _background_server as background_server
from runtime_session.session import RuntimeSession
from runtime_session.store import STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import ExecutionConfig, ModelControlConfig, SessionState, StateMessage, TeleopControlConfig
from settings import RerunSettings

from .fixtures import make_environment
from .test_model import _FarModel


@pytest.fixture
def store():
    store = Store.create(FeatureSpec(("shoulder", "gripper"), {"wrist": ImageSpec(4, 4)}), uuid4().hex)
    yield store
    store.unlink()


def _paths(log: MagicMock) -> list[str]:
    return [call.args[0] for call in log.call_args_list]


def _tick(*, step: int = 0, timestamp: float | None = None) -> TickEvent:
    return TickEvent(
        session_id="callback",
        step=step,
        timestamp=timestamp if timestamp is not None else time.time(),
        robot_state=_Observation(np.array([1.0, 2.0]), time.monotonic()),
        camera_frames={},
        action_sent=np.array([3.0, 4.0]),
        loop_duration_s=0.01,
        sleep_time_s=0.04,
        stale_obs=False,
    )


@pytest.fixture
async def callback_observer(store: Store):
    observer = RerunObserver("callback", hz=20)
    observer.set_store(store)
    with patch("physicalai.runtime.callbacks.rerun.LocalSystemMonitor") as monitor:
        monitor.return_value.poll.return_value = None
        async with observer.lifecycle():
            yield observer


@pytest.fixture
async def model_session(new_session):
    session = await new_session()
    await session.load_environment(make_environment(cameras=0))

    def build(config, store, _leader, _model, _bus, *, observer):
        return ModelControl(store, _FarModel(), config, observer=observer)

    with patch("runtime_session.session.build_control", side_effect=build):
        yield session
        await session.stop()


async def test_spawn_uses_explicit_port_and_a_recording_per_session() -> None:
    with patch("rerun.spawn") as spawn:
        settings = RerunSettings(open_viewer=True)
        async with (
            RerunObserver("first", settings=settings) as first,
            RerunObserver("second", settings=settings) as second,
        ):
            assert first._recording is not second._recording
            assert rr.get_recording_id(first._recording) == "first"
            assert rr.get_recording_id(second._recording) == "second"
            assert rr.get_global_data_recording() is None
        assert [call.kwargs["port"] for call in spawn.call_args_list] == [9876, 9876]


async def test_runtime_callback_routes_each_threads_logs_to_its_own_recording() -> None:
    seen = {}
    global_recording = rr.get_global_data_recording()

    def capture(_callback, event):
        seen[event.session_id] = rr.get_recording_id()

    with patch("rerun.init") as init, patch.object(RerunCallback, "on_tick", autospec=True, side_effect=capture):
        async with RerunObserver("first") as first, RerunObserver("second") as second:
            for observer in (first, second):
                observer.control_events(30, 0).emit_tick(replace(_tick(), session_id=observer.session_id))
            async with asyncio.timeout(5):
                while len(seen) < 2:  # noqa: ASYNC110
                    await asyncio.sleep(0.02)
            assert seen == {"first": "first", "second": "second"}
            assert rr.get_global_data_recording() is global_recording
        init.assert_not_called()


async def test_runtime_callback_supplies_joint_styles_and_blueprint(store: Store) -> None:
    observer = RerunObserver("styles", hz=50)
    observer.set_store(store)
    with patch("rerun.log") as log, patch("rerun.send_blueprint") as blueprint:
        async with observer.lifecycle():
            assert observer._callback._fps == 50
            assert observer._callback._joint_names == ["shoulder", "gripper"]
            assert observer._callback._camera_names == ["wrist"]
            styles = {call.args[0]: call.args[1] for call in log.call_args_list if call.kwargs.get("static")}
            assert isinstance(styles["robot/actions"], rr.SeriesLines)
            assert isinstance(styles["robot/predicted"], rr.SeriesPoints)
            assert isinstance(styles["robot/joints"], rr.SeriesLines)
            blueprint.assert_called_once()
            assert blueprint.call_args.kwargs == {"make_active": True, "make_default": True}


async def test_default_starts_headless_and_connect_mode_never_spawns() -> None:
    with patch("rerun.spawn") as spawn, patch("rerun.RecordingStream.connect_grpc") as connect:
        async with RerunObserver("background"):
            connect.assert_called_with("rerun+http://127.0.0.1:9876/proxy")
        async with RerunObserver("remote", settings=RerunSettings(mode="connect", connect_addr="host:1234")):
            connect.assert_called_with("rerun+http://host:1234/proxy")
        spawn.assert_not_called()


def test_real_headless_server_is_shared_and_reaped_after_the_last_session() -> None:
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    with background_server(port):
        process = _BACKGROUND_SERVERS[port].process
        assert process.poll() is None
        assert _listening(port)
        response = httpx.options(
            f"http://127.0.0.1:{port}/rerun.sdk_comms.v1.MessageProxyService/ReadMessages",
            headers={
                "Origin": "https://app.rerun.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-grpc-web,x-user-agent",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://app.rerun.io"
        with background_server(port):
            assert _BACKGROUND_SERVERS[port].process is process
            assert _BACKGROUND_SERVERS[port].users == 2
        assert process.poll() is None
    assert process.poll() is not None
    assert port not in _BACKGROUND_SERVERS
    assert not _listening(port)


def test_an_existing_server_is_borrowed_not_started_or_stopped() -> None:
    with patch("runtime_session.rerun._listening", return_value=True), patch("subprocess.Popen") as start:
        with background_server(12345):
            assert 12345 not in _BACKGROUND_SERVERS
        start.assert_not_called()


async def test_disabled_logging_does_not_start_an_observer(model_session, monkeypatch, tmp_path) -> None:
    from settings import merge_user_settings

    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    merge_user_settings({"rerun": {"enabled": False}})
    with patch("runtime_session.session.RerunObserver") as observer:
        session = model_session
        await session.set_control(ModelControlConfig())
    observer.assert_not_called()
    assert session._observer is None
    assert session.state().rerun_url is None


@pytest.mark.parametrize(
    ("settings", "stream_url"),
    [
        (RerunSettings(spawn_port=1234), "rerun+http://127.0.0.1:1234/proxy"),
        (RerunSettings(mode="connect", connect_addr="host:4321"), "rerun+http://host:4321/proxy"),
        (
            RerunSettings(mode="connect", connect_addr="rerun+https://host:4321/proxy"),
            "rerun+https://host:4321/proxy",
        ),
    ],
)
def test_browser_link_matches_sdk_version_and_session_connection(settings: RerunSettings, stream_url: str) -> None:
    observer = RerunObserver("browser", settings=settings)
    with patch("subprocess.Popen") as spawn:
        url = urlsplit(observer.viewer_url)
    spawn.assert_not_called()
    assert url.scheme == "https"
    assert url.netloc == "app.rerun.io"
    assert url.path == f"/version/{rr.__version__}/"
    assert parse_qs(url.query) == {"url": [stream_url]}


async def test_session_viewer_link_keeps_its_startup_settings(model_session, monkeypatch, tmp_path) -> None:
    from settings import merge_user_settings

    monkeypatch.setenv("SETTINGS_FILE", str(tmp_path / "settings.json"))
    merge_user_settings({"rerun": {"spawn_port": 1234}})
    session = model_session
    await session.set_control(ModelControlConfig())
    initial_url = session.state().rerun_url
    assert parse_qs(urlsplit(initial_url).query)["url"] == ["rerun+http://127.0.0.1:1234/proxy"]
    merge_user_settings({"rerun": {"spawn_port": 4321}})
    assert session.state().rerun_url == initial_url
    await session.set_control(None)
    assert session.state().rerun_url is None
    await session.set_control(ModelControlConfig())
    assert parse_qs(urlsplit(session.state().rerun_url).query)["url"] == ["rerun+http://127.0.0.1:4321/proxy"]


async def test_runtime_callback_logs_ticks_and_caps_latest_images(callback_observer, store: Store) -> None:
    observer = callback_observer
    now = time.monotonic()
    store.write_observation("wrist", np.zeros((4, 4, 3), dtype=np.uint8), timestamp=now)
    events = observer.control_events(20, 0)
    with patch("rerun.log") as log:
        events.emit_tick(_tick())
        observer._drain_events()
        assert "robot/joints" in _paths(log)
        assert "robot/actions" in _paths(log)
        assert "camera/wrist" in _paths(log)
        assert "runtime/loop_duration_s" in _paths(log)
        log.reset_mock()

        store.write_observation("wrist", np.ones((4, 4, 3), dtype=np.uint8), timestamp=now + 0.01)
        events.emit_tick(_tick(step=1))
        observer._drain_events()
        assert "robot/joints" in _paths(log)
        assert "camera/wrist" not in _paths(log)

        observer._next_images = 0
        events.emit_tick(_tick(step=2))
        observer._drain_events()
        assert "camera/wrist" in _paths(log)


async def test_detaching_releases_store_before_unlink(callback_observer, store: Store) -> None:
    observer = callback_observer
    store.write_observation("wrist", np.zeros((4, 4, 3), dtype=np.uint8), timestamp=time.monotonic())
    frames = observer._camera_frames()
    observer.set_store(None)
    store.write_observation("wrist", np.ones((4, 4, 3), dtype=np.uint8), timestamp=time.monotonic())
    assert not frames["wrist"].data.any()  # No shared-memory view escapes the copy.
    with patch.object(store, "read_observation", side_effect=AssertionError("detached store was read")):
        observer._next_images = 0
        assert observer._camera_frames() == {}


async def test_prediction_offset_rate_and_grace_period(callback_observer) -> None:
    observer = callback_observer
    assert isinstance(observer._callback, RerunCallback)
    events = observer.control_events(hz=20, grace_s=5)
    chunk = np.array([[1, 2], [3, 4], [5, 6]])
    now = time.time()
    events.emit_tick(_tick(step=10, timestamp=now))
    events.emit_inference(InferenceEvent("callback", now, latency_s=0.1, offset=1, chunk=chunk))
    chunk[:] = -1  # Executor-owned arrays must not change a queued visualization.
    prediction = observer._events[-1]
    assert isinstance(prediction, _Prediction)
    np.testing.assert_array_equal(prediction.event.chunk, [[3, 4], [5, 6]])
    events.emit_metrics(MetricsEvent("infer", 0, time.time(), {"queue_remaining": 2}))
    with patch("rerun.TimeColumn", wraps=rr.TimeColumn) as column, patch("rerun.log") as log:
        observer._drain_events()
    np.testing.assert_allclose(column.call_args_list[0].kwargs["sequence"], [110, 111])
    np.testing.assert_allclose(column.call_args_list[1].kwargs["timestamp"], events.ready_at + np.arange(2) / 20)
    assert observer._callback._last_step == 10
    assert "inference/latency_ms" in _paths(log)
    assert "queue/remaining" in _paths(log)
    assert "runtime/monitoring" in _paths(log)
    assert list(observer._callback._latencies) == [0.1]


@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_model_control_emits_predictions_ticks_and_queue_depth(
    callback_observer, store: Store, mode: str
) -> None:
    observer = callback_observer
    store.write(STATE, [1, 2], timestamp=time.monotonic())
    config = ModelControlConfig(hz=100, grace_s=0, execution=ExecutionConfig(mode=mode))
    async with ModelControl(store, _FarModel(), config, observer=observer):  # type: ignore[arg-type]
        async with asyncio.timeout(5):
            while not any(isinstance(item, _Prediction) for item in observer._events):  # noqa: ASYNC110
                await asyncio.sleep(0.02)
    ticks = [event for event in observer._events if isinstance(event, TickEvent)]
    assert ticks and ticks[0].step == 0
    assert ticks[0].robot_state.joint_positions.tolist() == [1, 2]
    assert ticks[0].action_sent is not None
    with patch("rerun.log") as log:
        observer._drain_events()
    assert "inference/latency_ms" in _paths(log)
    assert "queue/remaining" in _paths(log)
    assert "robot/actions" in _paths(log)


@pytest.mark.parametrize("grace_remaining", [0, 0.5])
async def test_slow_sync_inference_aligns_predictions_to_the_actual_control_step(
    callback_observer, store: Store, grace_remaining: float
) -> None:
    observer = callback_observer
    store.write(STATE, [1, 2], timestamp=time.monotonic())
    control = ModelControl(store, _FarModel(), ModelControlConfig(hz=20), observer=observer)
    control.on_start()
    control.step = 7
    observer._callback.on_tick(_tick(step=6, timestamp=100))
    assert control._events is not None
    control._events.ready_at = 105 + grace_remaining

    def slow_update(*_args):
        # Five seconds have elapsed since the last tick, but only one step runs.
        control._events.emit_inference(InferenceEvent("callback", 105, 5, 0, np.zeros((2, 2))))
        return np.array([3.0, 4.0])

    try:
        with patch.object(control.source, "update", side_effect=slow_update):
            control.on_tick()
        with patch("rerun.TimeColumn", wraps=rr.TimeColumn) as column:
            observer._drain_events()
        start_step = 7 + max(0, round(grace_remaining * 20) - 1)
        np.testing.assert_array_equal(column.call_args_list[0].kwargs["sequence"], [start_step, start_step + 1])
        assert observer._callback._last_step == 7
    finally:
        control.on_stop()


@pytest.mark.parametrize("mode", ["connect", "native"])
async def test_connect_and_native_runs_lease_a_studio_owned_server(mode: str) -> None:
    settings = RerunSettings(mode="connect") if mode == "connect" else RerunSettings(open_viewer=True)
    process = MagicMock()
    with (
        patch("runtime_session.rerun._background_server", background_server),
        patch("runtime_session.rerun._listening", side_effect=[False, True]),
        patch("subprocess.Popen", return_value=process) as spawn,
    ):
        with ExitStack() as headless:
            headless.enter_context(background_server(9876))
            async with RerunObserver("lease", settings=settings):
                assert _BACKGROUND_SERVERS[9876].users == 2
                headless.close()
                assert _BACKGROUND_SERVERS[9876].users == 1
                process.terminate.assert_not_called()
        process.terminate.assert_called_once_with()
        assert 9876 not in _BACKGROUND_SERVERS
        spawn.assert_called_once()


@pytest.mark.parametrize("address", ["127.0.0.1:9876", "localhost:9876", "remote:9876"])
async def test_connect_mode_never_starts_or_owns_an_external_server(address: str) -> None:
    with (
        patch("runtime_session.rerun._background_server", background_server),
        patch("runtime_session.rerun._listening", return_value=False),
        patch("subprocess.Popen") as spawn,
    ):
        async with RerunObserver("external", settings=RerunSettings(mode="connect", connect_addr=address)):
            assert 9876 not in _BACKGROUND_SERVERS
        spawn.assert_not_called()


def test_background_server_uses_the_deployment_bind_host(monkeypatch) -> None:
    monkeypatch.setenv("RERUN_BIND_HOST", "0.0.0.0")
    with (
        patch("runtime_session.rerun._listening", side_effect=[False, True]),
        patch("subprocess.Popen") as spawn,
        background_server(12345),
    ):
        argv = spawn.call_args.args[0]
        assert argv[argv.index("--bind") + 1] == "0.0.0.0"
        assert argv[argv.index("--port") + 1] == "12345"


def test_compose_publishes_rerun_only_on_host_loopback() -> None:
    from pathlib import Path

    import yaml

    compose = yaml.safe_load((Path(__file__).parents[3] / "docker/docker-compose.yaml").read_text())
    for service in compose["services"].values():
        assert "127.0.0.1:${RERUN_PORT:-9876}:${RERUN_PORT:-9876}" in service["ports"]
        assert "RERUN_BIND_HOST=0.0.0.0" in service["environment"]


def test_event_buffer_is_bounded() -> None:
    observer = RerunObserver("bounded")
    events = observer.control_events(30, 0)
    for step in range(1000):
        events.emit_metrics(MetricsEvent("bounded", step, time.time(), {"queue_remaining": step}))
    assert len(observer._events) == 64
    assert observer._events[-1].step == 999


async def test_session_events_are_logged_alongside_runtime_events(callback_observer) -> None:
    observer = callback_observer
    observer.emit_session(StateMessage(data=SessionState(session_id="hold")))
    with patch("rerun.log") as log:
        observer._drain_events()
    assert _paths(log) == ["runtime/session"]


async def test_runtime_callback_owns_system_gpu_monitoring_and_cleanup(callback_observer) -> None:
    observer = callback_observer
    monitor = observer._callback._system_monitor
    monitor.reset.assert_called_once_with()
    monitor.poll.return_value = SystemStatsSample(
        system=SystemStats(25, 50, 8 * 2**30, 16 * 2**30, 10, 100 * 2**20),
        gpu=GpuStatsSample(devices=[GpuDeviceStats("Intel", "0", 75, 512, 2048, 40)]),
    )
    observer.control_events(20, 0).emit_tick(_tick())
    with patch("rerun.log") as log:
        observer._drain_events()
    assert "runtime/utilization/cpu/host" in _paths(log)
    assert "runtime/utilization/cpu/process" in _paths(log)
    assert "runtime/utilization/gpu/intel_0" in _paths(log)
    assert "runtime/monitoring" in _paths(log)


async def test_callback_cleanup_and_disconnect_run_even_when_logging_fails() -> None:
    observer = RerunObserver("cleanup")
    with (
        patch("physicalai.runtime.callbacks.rerun.LocalSystemMonitor") as monitor,
        patch("rerun.RecordingStream.disconnect") as disconnect,
        patch.object(RerunCallback, "on_tick", side_effect=RuntimeError("logging failed")),
    ):
        with pytest.raises(RuntimeError, match="logging failed"):
            async with observer.lifecycle():
                observer.control_events(30, 0).emit_tick(_tick())
        monitor.return_value.close.assert_called_once_with()
        disconnect.assert_called_once_with()


async def test_real_sdk_accepts_blueprint_images_and_prediction_columns(store: Store) -> None:
    observer = RerunObserver("sdk")
    observer.set_store(store)
    async with observer.lifecycle():
        stream = observer._recording.binary_stream()
        store.write_observation("wrist", np.zeros((4, 4, 3), dtype=np.uint8), timestamp=time.monotonic())
        rr.set_strict_mode(True)
        try:
            observer.control_events(30, 0).emit_tick(_tick())
            observer.control_events(30, 0).emit_inference(InferenceEvent("sdk", time.time(), 0.01, 0, np.zeros((3, 2))))
            observer.control_events(30, 0).emit_inference(InferenceEvent("sdk", time.time(), 0.02, 3, np.zeros((3, 2))))
            observer._drain_events()
        finally:
            rr.set_strict_mode(False)
        assert stream.read(flush=True)


async def test_viewer_start_failure_does_not_prevent_session_use(model_session) -> None:
    session = model_session
    with patch("runtime_session.rerun._background_server", side_effect=RuntimeError("cannot start")):
        await session.set_control(ModelControlConfig())
    assert session._observer is None
    assert session.state().control.type == "model"
    assert session.state().session_id == session.id
    await session.stop()
    assert session.finished.is_set()


async def test_session_joins_observer_on_close(model_session) -> None:
    session = model_session
    await session.set_control(ModelControlConfig())
    observer = session._observer
    assert observer._store is session.store
    await session.stop()
    assert observer._store is None
    assert session._observer is None
    assert not observer.is_alive()


async def test_logging_failure_does_not_close_the_session(model_session) -> None:
    session = model_session
    await session.set_control(ModelControlConfig())
    observer = session._observer
    with patch.object(observer._callback, "on_tick", side_effect=RuntimeError("viewer failed")):
        async with asyncio.timeout(5):
            await observer.wait_stopped()
    assert observer.failure is not None
    assert not session.closed.is_set()
    await session.stop()


async def test_opening_session_loading_environment_and_teleop_do_not_start_rerun(new_session) -> None:
    with patch("runtime_session.session.RerunObserver") as observer:
        session = await new_session()
        await session.load_environment(make_environment(cameras=0))
        await session.set_control(TeleopControlConfig())
        await session.set_control(None)
        await session.stop()
    observer.assert_not_called()
    assert session.state().rerun_url is None


async def test_play_stop_play_creates_separate_recordings(model_session: RuntimeSession) -> None:
    session = model_session
    initial = session._observer
    assert initial is None
    await session.set_control(ModelControlConfig())
    first = session._observer
    assert first is not None
    assert first.is_alive()
    assert first._store is session.store
    first_id = rr.get_recording_id(first._recording)
    assert session.state().rerun_url is not None

    await session.set_control(None)
    assert not first.is_alive()
    stopped = session._observer
    assert stopped is None
    assert session.state().rerun_url is None

    await session.set_control(ModelControlConfig())
    second = session._observer
    assert second is not None
    assert second is not first
    assert second.is_alive()
    assert rr.get_recording_id(second._recording) != first_id


async def test_control_start_failure_stops_recording(model_session) -> None:
    session = model_session
    observers = []

    def refuse(*_args, observer, **_kwargs):
        observers.append(observer)
        raise ValueError("cannot start control")

    with (
        patch("runtime_session.session.build_control", side_effect=refuse),
        pytest.raises(ValueError, match="cannot start control"),
    ):
        await session.set_control(ModelControlConfig())
    assert len(observers) == 1
    assert not observers[0].is_alive()
    assert session._observer is None
    assert session.state().rerun_url is None


@pytest.mark.parametrize("command", ["unload_environment", "reload_environment", "unload_model"])
async def test_unloading_or_reloading_stops_recording(model_session, command: str) -> None:
    session = model_session
    await session.set_control(ModelControlConfig())
    observer = session._observer
    await getattr(session, command)()
    assert not observer.is_alive()
    assert session._observer is None

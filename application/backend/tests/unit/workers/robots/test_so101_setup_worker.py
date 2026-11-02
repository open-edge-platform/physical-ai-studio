# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Protocol and deterministic cancellation/range tests with no serial devices."""

import asyncio
import copy
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from physicalai_so101_plugin import SO101Calibration
from physicalai_so101_plugin.constants import SO101_JOINT_ORDER

from schemas import SerialPortInfo
from workers.robots import so101_setup_worker as module
from workers.robots.so101_setup_worker import SetupPhase, SO101SetupWorker
from workers.transport_worker import WorkerState


def make_calibration():
    return SO101Calibration.from_dict(
        {
            name: {"id": servo_id, "drive_mode": 0, "homing_offset": -100, "range_min": 100, "range_max": 3100}
            for servo_id, name in enumerate(SO101_JOINT_ORDER, 1)
        }
    )


class FakeSession:
    def __init__(self, port="fake", *, role="follower"):
        self.port, self.role = port, role
        self.servo_ids = dict(zip(SO101_JOINT_ORDER, range(1, 7), strict=True))
        self.calibration = None
        self.positions = dict.fromkeys(self.servo_ids, 2047)
        self.models = dict.fromkeys(self.servo_ids, 777)
        self.voltages = dict.fromkeys(self.servo_ids, 120)
        self.calls = []
        self.written = None
        self.failure = None
        self.entered = None
        self.release = None
        self.active = False
        self.closed = False

    def call(self, name):
        assert not self.active, "Overlapping hardware operations"
        self.calls.append(name)
        if self.failure == name:
            raise RuntimeError(f"Injected {name} failure")

    def connect(self):
        self.call("connect")

    def disconnect(self):
        self.call("disconnect")
        self.closed = True

    def ping(self, name):
        self.call("ping")
        return self.models[name]

    def read_voltage(self, name):
        self.call("read_voltage")
        value = self.voltages[name]
        if value is None:
            raise ConnectionError("Missing voltage")
        return value

    def disable_torque(self):
        self.call("disable_torque")

    def setup_motor(self, name):
        self.call("setup_motor")

    def clear_calibration(self):
        self.call("clear_calibration")
        self.calibration = None

    def set_half_turn_homings(self):
        self.call("homing")
        return dict.fromkeys(self.servo_ids, -100)

    def read_positions(self, *, normalized=False):
        self.call("read_positions")
        self.active = True
        try:
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(5)
            return dict.fromkeys(self.servo_ids, 50.0) if normalized else dict(self.positions)
        finally:
            self.active = False

    def read_calibration(self):
        self.call("read_calibration")
        return make_calibration()

    def load_calibration(self, calibration):
        self.call("load_calibration")
        self.calibration = calibration

    def write_calibration(self, calibration):
        self.call("write_calibration")
        self.active = True
        try:
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(5)
            self.written = copy.deepcopy(calibration.to_dict())
            self.calibration = calibration
        finally:
            self.active = False

    def configure_motors(self):
        self.call("configure_motors")


@pytest.fixture
def worker(monkeypatch):
    transport = SimpleNamespace(
        connect=AsyncMock(), send_json=AsyncMock(), close=AsyncMock(), receive_command=AsyncMock()
    )
    worker = SO101SetupWorker(transport, "SO101_Follower", SerialPortInfo(connection_string="fake"), object())
    worker.setup_session = FakeSession()
    worker.probe_result = {"all_motors_ok": True}
    worker.phase = SetupPhase.MOTOR_PROBE
    worker.state = WorkerState.RUNNING
    return worker


def events(worker, event):
    return [call.args[0] for call in worker.transport.send_json.call_args_list if call.args[0]["event"] == event]


async def command(worker, name, **kwargs):
    await worker._dispatch_command(name, kwargs)


@pytest.mark.asyncio
async def test_full_flow_keeps_follower_released_and_payloads_unchanged(worker):
    session = worker.setup_session
    await command(worker, "enter_calibration")
    assert session.calls[-1] == "disable_torque"
    assert worker.phase == SetupPhase.CALIBRATION_INSTRUCTIONS
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    session.positions = dict.fromkeys(session.servo_ids, 100)
    await worker._broadcast_tick()
    session.positions = dict.fromkeys(session.servo_ids, 3100)
    await worker._broadcast_tick()
    await command(worker, "stop_recording")
    assert worker.phase == SetupPhase.VERIFICATION
    assert session.written == make_calibration().to_dict()
    assert events(worker, "calibration_result") == [{"event": "calibration_result", "calibration": session.written}]
    assert session.calls.index("write_calibration") < session.calls.index("configure_motors")
    assert session.calls[-1] == "disable_torque"
    await worker._broadcast_tick()
    assert events(worker, "state_was_updated")[-1] == {
        "event": "state_was_updated",
        "state": {f"{name}.pos": 50.0 for name in session.servo_ids},
    }
    await command(worker, "enter_verification")  # UI repeats after automatic transition.
    await command(worker, "enter_calibration")  # Back/recalibrate is legitimate.
    assert worker.homing_offsets is None
    assert not worker._range_mins
    assert session.calibration is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,raw,expected",
    [
        ("SO101_Follower", 70, True),
        ("SO101_Follower", 69, False),
        ("SO101_Leader", 69, True),
        ("SO101_Leader", 70, False),
        ("SO101_Follower", None, True),
        ("SO101_Leader", 0, True),
    ],
)
async def test_voltage_policy_and_unknown_values(worker, role, raw, expected):
    worker.robot_type = role
    worker.setup_session.voltages = dict.fromkeys(SO101_JOINT_ORDER, raw)
    async with worker._operation_lock:
        await worker._run_voltage_check()
    payload = events(worker, "voltage_result")[-1]
    assert set(payload) == {"event", "readings", "avg_voltage", "voltage_ok", "expected_source", "robot_type"}
    assert payload["voltage_ok"] is expected
    assert payload["avg_voltage"] == (None if raw is None else raw * 0.1)
    assert payload["readings"][0] == {"name": "shoulder_pan", "motor_id": 1, "raw": raw}


@pytest.mark.asyncio
async def test_missing_and_wrong_model_probe(worker):
    worker.setup_session.models["gripper"] = None
    worker.setup_session.models["wrist_roll"] = 2825
    async with worker._operation_lock:
        await worker._run_motor_probe()
    payload = events(worker, "motor_probe_result")[-1]
    assert payload["found_count"] == 4
    assert payload["total_count"] == 6
    assert payload["calibration"] is None
    assert not payload["all_motors_ok"]
    with pytest.raises(ValueError, match="six"):
        await command(worker, "enter_calibration")


@pytest.mark.asyncio
async def test_skipped_calibration_reads_loads_releases_without_writes(worker):
    await command(worker, "enter_verification")
    assert worker.setup_session.calls == ["disable_torque", "read_calibration", "load_calibration"]
    assert events(worker, "calibration_result")[-1]["calibration"] == make_calibration().to_dict()
    await command(worker, "enter_verification")
    assert worker.setup_session.calls.count("load_calibration") == 1


@pytest.mark.asyncio
async def test_motor_setup_events_and_stale_state_reset(worker):
    worker.setup_session.calibration = make_calibration()
    worker.homing_offsets = {"gripper": 1}
    await command(worker, "start_motor_setup")
    assert worker.probe_result is None and worker.homing_offsets is None
    assert worker.setup_session.calibration is None
    with pytest.raises(ValueError, match="Unknown motor"):
        await command(worker, "motor_connected", motor="unknown")
    await command(worker, "motor_connected", motor="gripper")
    assert [event["status"] for event in events(worker, "motor_setup_progress")] == ["scanning", "success"]
    worker.setup_session.failure = "setup_motor"
    await command(worker, "motor_connected", motor="gripper")
    assert events(worker, "motor_setup_progress")[-1]["status"] == "error"
    worker.setup_session.failure = None
    await command(worker, "finish_motor_setup")
    assert worker.probe_result["all_motors_ok"]


@pytest.mark.asyncio
async def test_recording_prerequisites_and_zero_span_retry(worker):
    with pytest.raises(ValueError):
        await command(worker, "start_recording")
    with pytest.raises(ValueError):
        await command(worker, "stop_recording")
    await command(worker, "enter_calibration")
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    with pytest.raises(ValueError, match="min/max"):
        await command(worker, "stop_recording")
    assert worker.phase == SetupPhase.CALIBRATION_RECORDING
    worker.setup_session.positions["gripper"] = -1
    before = dict(worker._range_mins)
    with pytest.raises(ValueError, match="encoder"):
        await worker._broadcast_tick()
    assert worker._range_mins == before
    await command(worker, "enter_calibration")
    assert worker.phase == SetupPhase.CALIBRATION_INSTRUCTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write_calibration", "configure_motors"])
async def test_failed_write_or_configuration_has_no_success_or_verification(worker, failure):
    await command(worker, "enter_calibration")
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    worker._range_mins = dict.fromkeys(SO101_JOINT_ORDER, 100)
    worker._range_maxes = dict.fromkeys(SO101_JOINT_ORDER, 3100)
    worker.setup_session.failure = failure
    with pytest.raises(RuntimeError):
        await command(worker, "stop_recording")
    assert worker.phase == SetupPhase.CALIBRATION_RECORDING
    assert worker.setup_session.calibration is None
    assert not events(worker, "calibration_result")
    with pytest.raises(ValueError):
        await command(worker, "enter_verification")


@pytest.mark.asyncio
async def test_raw_change_detection_and_phase_matrix(worker):
    for phase in [SetupPhase.MOTOR_SETUP, SetupPhase.MOTOR_PROBE, SetupPhase.CONFIGURE, SetupPhase.CONNECTING]:
        worker._set_phase(phase)
        await worker._broadcast_tick()
    assert not worker.setup_session.calls
    worker._set_phase(SetupPhase.CALIBRATION_INSTRUCTIONS)
    await worker._broadcast_tick()
    await worker._broadcast_tick()
    assert len(events(worker, "positions")) == 1
    worker._set_phase(SetupPhase.CALIBRATION_HOMING)
    await worker._broadcast_tick()
    assert len(events(worker, "positions")) == 2


@pytest.mark.asyncio
async def test_stale_read_waiting_on_lock_is_not_started(worker):
    worker._set_phase(SetupPhase.CALIBRATION_INSTRUCTIONS)
    await worker._operation_lock.acquire()
    tick = asyncio.create_task(worker._broadcast_tick())
    await asyncio.sleep(0)  # Tick has captured its phase and is queued on lock.
    worker._set_phase(SetupPhase.CONFIGURE)
    worker._operation_lock.release()
    await tick
    assert not worker.setup_session.calls
    assert not events(worker, "positions")


@pytest.mark.asyncio
async def test_cancelled_thread_is_drained_before_cleanup_and_no_late_frame(worker):
    worker._set_phase(SetupPhase.CALIBRATION_INSTRUCTIONS)
    session = worker.setup_session
    session.entered, session.release = threading.Event(), threading.Event()
    tick = asyncio.create_task(worker._broadcast_tick())
    assert await asyncio.to_thread(session.entered.wait, 5)
    tick.cancel()
    await asyncio.sleep(0)
    tick.cancel()  # Repeated owner cancellation cannot release serial ownership.
    cleanup = asyncio.create_task(worker._cleanup())
    await asyncio.sleep(0)
    assert not session.closed and worker._operation_lock.locked()
    session.release.set()
    with pytest.raises(asyncio.CancelledError):
        await tick
    await cleanup
    assert session.closed and worker.setup_session is None
    assert not events(worker, "positions")


@pytest.mark.asyncio
async def test_stop_waits_for_accepted_sample_and_freezes_snapshot(worker):
    await command(worker, "enter_calibration")
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    session = worker.setup_session
    session.positions = dict.fromkeys(SO101_JOINT_ORDER, 3100)
    session.entered, session.release = threading.Event(), threading.Event()
    tick = asyncio.create_task(worker._broadcast_tick())
    assert await asyncio.to_thread(session.entered.wait, 5)
    stopped = asyncio.create_task(command(worker, "stop_recording"))
    await asyncio.sleep(0)
    session.release.set()
    await asyncio.gather(tick, stopped)
    assert all(cal["range_max"] == 3100 for cal in session.written.values())
    frozen = copy.deepcopy(session.written)
    worker._range_maxes["gripper"] = 4095
    assert session.written == frozen


@pytest.mark.asyncio
async def test_run_closes_resources_even_when_error_reporting_fails(worker, monkeypatch):
    session = worker.setup_session
    monkeypatch.setattr(module, "SO101SetupSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(module, "find_so101_port", AsyncMock(return_value="fake"))
    session.failure = "connect"
    worker.transport.send_json.side_effect = [None, RuntimeError("websocket gone")]
    await worker.run()
    assert session.closed
    worker.transport.close.assert_awaited_once()


def test_worker_import_does_not_load_lerobot_motors():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class RejectMotors(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'lerobot.motors' or fullname.startswith('lerobot.motors.'):
            raise AssertionError('LeRobot motor import: ' + fullname)
sys.meta_path.insert(0, RejectMotors())
from workers.robots.so101_setup_worker import SO101SetupWorker
assert not any(name.startswith('lerobot.motors') for name in sys.modules)
""",
        ],
        check=True,
    )


@pytest.mark.asyncio
async def test_cancelled_calibration_write_drains_and_does_not_start_configuration(worker):
    await command(worker, "enter_calibration")
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    worker._range_mins = dict.fromkeys(SO101_JOINT_ORDER, 100)
    worker._range_maxes = dict.fromkeys(SO101_JOINT_ORDER, 3100)
    session = worker.setup_session
    session.entered, session.release = threading.Event(), threading.Event()
    stopped = asyncio.create_task(command(worker, "stop_recording"))
    assert await asyncio.to_thread(session.entered.wait, 5)
    stopped.cancel()
    cleanup = asyncio.create_task(worker._cleanup())
    await asyncio.sleep(0)
    assert not session.closed and session.active
    session.release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopped
    await cleanup
    assert session.closed
    assert "configure_motors" not in session.calls
    assert not events(worker, "calibration_result")


@pytest.mark.asyncio
async def test_reprobe_cannot_bypass_failed_configuration(worker):
    await command(worker, "enter_calibration")
    await command(worker, "start_homing")
    await command(worker, "start_recording")
    worker._range_mins = dict.fromkeys(SO101_JOINT_ORDER, 100)
    worker._range_maxes = dict.fromkeys(SO101_JOINT_ORDER, 3100)
    worker.setup_session.failure = "configure_motors"
    with pytest.raises(RuntimeError):
        await command(worker, "stop_recording")
    worker.setup_session.failure = None
    await command(worker, "re_probe")
    with pytest.raises(ValueError, match="Complete homing"):
        await command(worker, "enter_verification")


@pytest.mark.asyncio
async def test_streaming_uses_30fps_target_without_wall_clock_sleep(worker, monkeypatch):
    times = iter([10.0, 10.01])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    worker._broadcast_tick = AsyncMock()
    sleeps = []

    async def sleep(interval):
        sleeps.append(interval)
        worker._stop_requested = True

    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    await worker._broadcast_loop()
    assert sleeps == pytest.approx([1 / 30 - 0.01])

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SO101 wizard orchestration; the first-party setup session owns hardware I/O."""

import asyncio
import time
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from typing import Any, TypeVar

from loguru import logger
from physicalai_so101_plugin import SO101Calibration, SO101SetupSession
from physicalai_so101_plugin.constants import STS3215_MODEL_NUMBER
from physicalai_so101_plugin.studio_catalog import find_so101_port

from schemas import SerialPortInfo
from utils.serial_robot_tools import RobotConnectionManager
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState

VOLTAGE_THRESHOLD_RAW = 70
VOLTAGE_UNIT = 0.1
FPS = 30
T = TypeVar("T")


class SetupPhase(StrEnum):
    """Existing websocket phase values, unchanged by the hardware migration."""

    CONNECTING = "connecting"
    VOLTAGE_CHECK = "voltage_check"
    MOTOR_PROBE = "motor_probe"
    MOTOR_SETUP = "motor_setup"
    CALIBRATION_INSTRUCTIONS = "calibration_instructions"
    CALIBRATION_HOMING = "calibration_homing"
    CALIBRATION_RECORDING = "calibration_recording"
    CONFIGURE = "configure"
    VERIFICATION = "verification"


class SO101SetupWorker(TransportWorker):
    """Hold one setup session; preserve the wizard's commands and event schemas.

    The asyncio lock protects state transitions and accepted recording samples.
    Shielded, explicitly retained thread tasks keep cancellation from releasing
    a logical transaction while serial writes continue in another thread. The
    session's internal RLock independently protects synchronous consumers.
    """

    def __init__(
        self,
        transport: WorkerTransport,
        robot_type: str,
        serial_port: SerialPortInfo,
        robot_manager: RobotConnectionManager,
    ) -> None:
        super().__init__(transport)
        self.robot_type = robot_type
        self.serial_port = serial_port
        self.robot_manager = robot_manager
        self.phase = SetupPhase.CONNECTING
        self.setup_session: SO101SetupSession | None = None
        self.port: str | None = None
        self._operation_lock = asyncio.Lock()
        self._inflight: asyncio.Task | None = None
        self._generation = 0
        self._last_raw: dict[str, int] | None = None
        self._verification_ready = True  # An untouched EEPROM can be inspected/loaded for the skip path.
        self.voltage_result: dict[str, Any] | None = None
        self.probe_result: dict[str, Any] | None = None
        self.homing_offsets: dict[str, int] | None = None
        self._range_mins: dict[str, int] = {}
        self._range_maxes: dict[str, int] = {}

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, PermissionError):
            return "permission_denied"
        if isinstance(exc, ConnectionError):
            return "device_not_found"
        return "connection_failed"

    def _require_setup_session(self) -> SO101SetupSession:
        if self.setup_session is None:
            raise ConnectionError("SO101 setup session is not connected")
        return self.setup_session

    def _set_phase(self, phase: SetupPhase) -> None:
        self.phase = phase
        self._generation += 1
        self._last_raw = None

    def _require_phase(self, *phases: SetupPhase) -> None:
        if self.phase not in phases:
            raise ValueError(f"Command is unavailable during {self.phase.value}")

    async def _hardware_call(self, operation: Callable[..., T], *args: Any, cleanup: bool = False, **kwargs: Any) -> T:
        """Call only inside the operation lock; drain threads before cancellation.

        A timeout or repeated cancellation must never authorize another I/O
        submission, including disconnect, while this thread is still active.
        """
        if self._stop_requested and not cleanup:
            raise asyncio.CancelledError
        if self._inflight is not None:
            raise RuntimeError("SO101 hardware transaction already in progress")
        task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        self._inflight = task
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # Retrieve thread failures even when cancellation takes precedence.
            if not task.cancelled():
                task.exception()
            raise
        finally:
            self._inflight = None

    async def run(self) -> None:
        """Connect, diagnose, handle commands/streaming, then drain and close."""
        try:
            await self.transport.connect()
            self.state = WorkerState.RUNNING
            async with self._operation_lock:
                await self._connect_setup_session()
                await self._run_voltage_check()
                await self._run_motor_probe()
            await self._run_loops()
        except Exception as exc:
            self.state = WorkerState.ERROR
            self.error_message = str(exc)
            logger.exception("SO101 setup worker failed")
            try:
                await self._send_event("error", message=str(exc), error_code=self._classify_error(exc), port=self.port)
            except Exception:
                logger.debug("Could not report SO101 setup failure")
        finally:
            self._stop_requested = True
            self._generation += 1
            # Keep finalization alive even if the websocket owner cancels again.
            cleanup_task = asyncio.create_task(self._finalize())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            cleanup_task.result()

    async def _run_loops(self) -> None:
        # Base run_concurrent uses a 2s cancellation timeout. Setup transactions
        # can legitimately take longer; drain our tasks instead of timing out.
        tasks = [asyncio.create_task(self._broadcast_loop()), asyncio.create_task(self._command_loop())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self._stop_requested = True
            self._generation += 1
            for task in tasks:
                task.cancel()
            drain = asyncio.gather(*tasks, return_exceptions=True)
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    continue

    async def _finalize(self) -> None:
        try:
            await self._cleanup()
        finally:
            await self.shutdown()

    async def _connect_setup_session(self) -> None:
        self._set_phase(SetupPhase.CONNECTING)
        await self._send_phase_status("Resolving connection port...")
        port = await find_so101_port(self.robot_manager, self.serial_port)
        if port is None:
            if self.serial_port.serial_number:
                raise ConnectionError(f"No USB device found with serial number '{self.serial_port.serial_number}'")
            raise ConnectionError(f"No USB device found at '{self.serial_port.connection_string or ''}'")
        self.port = port
        role = "follower" if self.robot_type == "SO101_Follower" else "leader"
        self.setup_session = SO101SetupSession(port, role=role)
        try:
            await self._hardware_call(self.setup_session.connect)
        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied for port {port}. Add your user to the dialout group: sudo usermod -aG dialout $USER"
            ) from exc
        except OSError as exc:
            raise ConnectionError(f"Could not open port {port}: {exc}") from exc
        await self._send_phase_status(f"Connected to {port}")

    async def _run_voltage_check(self) -> None:
        self._set_phase(SetupPhase.VOLTAGE_CHECK)
        await self._send_phase_status("Checking supply voltage...")
        session = self._require_setup_session()
        readings = []
        values: list[int] = []
        for name, servo_id in session.servo_ids.items():
            raw = None
            try:
                raw = await self._hardware_call(session.read_voltage, name)
                values.append(raw)
            except Exception:
                logger.debug("Could not read SO101 voltage for {}", name)
            readings.append({"name": name, "motor_id": servo_id, "raw": raw})
        avg_raw = sum(values) / len(values) if values else None
        follower = self.robot_type == "SO101_Follower"
        voltage_ok = True
        if avg_raw is not None:
            voltage_ok = avg_raw >= VOLTAGE_THRESHOLD_RAW if follower else avg_raw < VOLTAGE_THRESHOLD_RAW
        self.voltage_result = {
            "event": "voltage_result",
            "readings": readings,
            "avg_voltage": None if avg_raw is None else avg_raw * VOLTAGE_UNIT,
            "voltage_ok": voltage_ok,
            "expected_source": "external power supply (>= 7V)" if follower else "USB only (< 7V)",
            "robot_type": self.robot_type,
        }
        await self.transport.send_json(self.voltage_result)

    async def _run_motor_probe(self) -> None:
        self._set_phase(SetupPhase.MOTOR_PROBE)
        await self._send_phase_status("Probing motors...")
        session = self._require_setup_session()
        motors = []
        for name, servo_id in session.servo_ids.items():
            model = await self._hardware_call(session.ping, name)
            motors.append(
                {
                    "name": name,
                    "motor_id": servo_id,
                    "found": model is not None,
                    "model_number": model,
                    "model_correct": model == STS3215_MODEL_NUMBER,
                }
            )
        all_ok = all(motor["found"] and motor["model_correct"] for motor in motors)
        calibration_status = None
        if all_ok:
            # Connecting intentionally does no writes. Release even a follower
            # left holding position by a previous deployment before manual setup.
            await self._hardware_call(session.disable_torque)
            try:
                calibration_status = await self._check_calibration()
            except (ValueError, ConnectionError, RuntimeError) as exc:
                # Invalid preexisting EEPROM must not prevent corrective homing.
                logger.warning("SO101 EEPROM calibration could not be inspected: {}", exc)
        self.probe_result = {
            "event": "motor_probe_result",
            "motors": motors,
            "all_motors_ok": all_ok,
            "found_count": sum(1 for motor in motors if motor["found"] and motor["model_correct"]),
            "total_count": len(session.servo_ids),
            "calibration": calibration_status,
        }
        await self.transport.send_json(self.probe_result)

    async def _check_calibration(self) -> dict[str, Any]:
        calibration = await self._hardware_call(self._require_setup_session().read_calibration)
        motors = {}
        for name, cal in calibration.joints.items():
            default = cal.homing_offset == 0 and cal.range_min == 0 and cal.range_max == 4095
            motors[name] = {
                "homing_offset": cal.homing_offset,
                "range_min": cal.range_min,
                "range_max": cal.range_max,
                "is_calibrated": not default,
            }
        return {"motors": motors, "all_calibrated": all(motor["is_calibrated"] for motor in motors.values())}

    async def _reset_calibration_flow(self) -> None:
        self._verification_ready = False
        self.homing_offsets = None
        self._range_mins = {}
        self._range_maxes = {}
        await self._hardware_call(self._require_setup_session().clear_calibration)

    async def _handle_motor_setup(self, name: str) -> None:
        self._require_phase(SetupPhase.MOTOR_SETUP)
        session = self._require_setup_session()
        if name not in session.servo_ids:
            raise ValueError(f"Unknown motor: {name}")
        await self._send_event(
            "motor_setup_progress", motor=name, status="scanning", message=f"Scanning for motor '{name}'..."
        )
        try:
            await self._hardware_call(session.setup_motor, name)
        except Exception as exc:
            await self._send_event("motor_setup_progress", motor=name, status="error", message=str(exc))
        else:
            await self._send_event(
                "motor_setup_progress",
                motor=name,
                status="success",
                message=f"Motor '{name}' configured as ID {session.servo_ids[name]}",
            )

    async def _enter_calibration(self) -> None:
        self._require_phase(
            SetupPhase.MOTOR_PROBE,
            SetupPhase.CALIBRATION_INSTRUCTIONS,
            SetupPhase.CALIBRATION_HOMING,
            SetupPhase.CALIBRATION_RECORDING,
            SetupPhase.CONFIGURE,
            SetupPhase.VERIFICATION,
        )
        self._require_good_probe()
        await self._reset_calibration_flow()
        await self._hardware_call(self._require_setup_session().disable_torque)
        self._set_phase(SetupPhase.CALIBRATION_INSTRUCTIONS)
        await self._send_phase_status("Torque disabled — support the arm while moving it freely.")

    def _require_good_probe(self) -> None:
        if self.probe_result is None or not self.probe_result["all_motors_ok"]:
            raise ValueError("Probe all six STS3215 motors successfully before calibration or verification")

    async def _handle_start_homing(self) -> None:
        self._require_phase(SetupPhase.CALIBRATION_INSTRUCTIONS, SetupPhase.CALIBRATION_HOMING)
        await self._reset_calibration_flow()
        self._set_phase(SetupPhase.CALIBRATION_HOMING)
        await self._send_phase_status("Applying homing offsets...")
        try:
            offsets = await self._hardware_call(self._require_setup_session().set_half_turn_homings)
        except Exception:
            self._set_phase(SetupPhase.CALIBRATION_INSTRUCTIONS)
            raise
        self.homing_offsets = dict(offsets)
        await self._send_event("homing_result", offsets=dict(offsets))

    @staticmethod
    def _recording_sample(positions: Mapping[str, int | float], names: Iterable[str]) -> dict[str, int]:
        if set(positions) != set(names):
            raise ValueError("Recording sample is missing expected joints")
        if any(type(value) is not int or not 0 <= value <= 4095 for value in positions.values()):
            raise ValueError("Invalid encoder sample during recording; positions must be integers in 0..4095")
        return {name: int(value) for name, value in positions.items()}

    async def _handle_start_recording(self) -> None:
        self._require_phase(SetupPhase.CALIBRATION_HOMING, SetupPhase.CALIBRATION_RECORDING)
        if self.homing_offsets is None:
            raise ValueError("Apply homing offsets before recording")
        if self.phase == SetupPhase.CALIBRATION_RECORDING:
            return
        session = self._require_setup_session()
        await self._hardware_call(session.disable_torque)
        positions = await self._hardware_call(session.read_positions, normalized=False)
        sample = self._recording_sample(positions, session.servo_ids)
        self._range_mins, self._range_maxes = dict(sample), dict(sample)
        self._set_phase(SetupPhase.CALIBRATION_RECORDING)
        await self._send_phase_status("Recording range of motion...")

    async def _handle_stop_recording(self) -> None:
        self._require_phase(SetupPhase.CALIBRATION_RECORDING)
        if self.homing_offsets is None or not self._range_mins or not self._range_maxes:
            raise ValueError("Start a homed range recording before stopping it")
        session = self._require_setup_session()
        # Copies freeze the accepted samples under the SAME lock as streaming.
        mins, maxes, offsets = dict(self._range_mins), dict(self._range_maxes), dict(self.homing_offsets)
        self._recording_sample(mins, session.servo_ids)
        self._recording_sample(maxes, session.servo_ids)
        invalid = [name for name in session.servo_ids if mins[name] >= maxes[name]]
        if invalid:
            raise ValueError(
                f"Some motors have the same or invalid min/max values: {invalid}. "
                "Move all joints through their full range."
            )
        calibration = SO101Calibration.from_dict(
            {
                name: {
                    "id": servo_id,
                    "drive_mode": 0,
                    "homing_offset": offsets[name],
                    "range_min": mins[name],
                    "range_max": maxes[name],
                }
                for name, servo_id in session.servo_ids.items()
            }
        )
        self._set_phase(SetupPhase.CONFIGURE)
        try:
            await self._hardware_call(session.write_calibration, calibration)
            await self._send_phase_status("Configuring motors...")
            await self._hardware_call(session.configure_motors)
        except Exception:
            # Configuration can fail after a successful EEPROM write. Do not
            # allow an explicit enter_verification to bypass that failed step.
            await self._hardware_call(session.clear_calibration)
            self._set_phase(SetupPhase.CALIBRATION_RECORDING)
            raise
        await self._send_phase_status("Calibration complete")
        self._verification_ready = True
        await self._send_event("calibration_result", calibration=calibration.to_dict())
        await self._enter_verification()

    async def _enter_verification(self) -> None:
        self._require_phase(SetupPhase.MOTOR_PROBE, SetupPhase.CONFIGURE, SetupPhase.VERIFICATION)
        self._require_good_probe()
        if not self._verification_ready:
            raise ValueError("Complete homing, range recording and configuration before verification")
        session = self._require_setup_session()
        await self._hardware_call(session.disable_torque)
        if session.calibration is None:
            calibration = await self._hardware_call(session.read_calibration)
            await self._hardware_call(session.load_calibration, calibration)
            await self._send_event("calibration_result", calibration=calibration.to_dict())
        self._set_phase(SetupPhase.VERIFICATION)

    async def _broadcast_loop(self) -> None:
        while not self._stop_requested:
            start = time.perf_counter()
            try:
                await self._broadcast_tick()
            except Exception as exc:
                logger.warning("SO101 setup streaming failed: {}", exc)
                await self._send_event("error", message=str(exc), error_code="command_error")
            await asyncio.sleep(max(0.001, 1.0 / FPS - (time.perf_counter() - start)))

    async def _broadcast_tick(self) -> None:
        phase, generation = self.phase, self._generation
        streaming = {
            SetupPhase.CALIBRATION_INSTRUCTIONS,
            SetupPhase.CALIBRATION_HOMING,
            SetupPhase.CALIBRATION_RECORDING,
            SetupPhase.VERIFICATION,
        }
        if phase not in streaming:
            return
        payload = None
        async with self._operation_lock:
            if self._stop_requested or (phase, generation) != (self.phase, self._generation):
                return
            session = self._require_setup_session()
            positions = await self._hardware_call(session.read_positions, normalized=phase == SetupPhase.VERIFICATION)
            if self._stop_requested or generation != self._generation:
                return
            if phase == SetupPhase.VERIFICATION:
                payload = {
                    "event": "state_was_updated",
                    "state": {f"{name}.pos": float(value) for name, value in positions.items()},
                }
            else:
                current = {name: int(value) for name, value in positions.items()}
                if phase == SetupPhase.CALIBRATION_RECORDING:
                    current = self._recording_sample(positions, session.servo_ids)
                    for name, position in current.items():
                        self._range_mins[name] = min(self._range_mins[name], position)
                        self._range_maxes[name] = max(self._range_maxes[name], position)
                    payload = {
                        "event": "positions",
                        "motors": {
                            name: {"position": position, "min": self._range_mins[name], "max": self._range_maxes[name]}
                            for name, position in current.items()
                        },
                    }
                elif current != self._last_raw:
                    payload = {
                        "event": "positions",
                        "motors": {name: {"position": value} for name, value in current.items()},
                    }
                self._last_raw = current
        if payload is not None and not self._stop_requested and generation == self._generation:
            await self.transport.send_json(payload)

    async def _command_loop(self) -> None:
        while not self._stop_requested:
            data = await self.transport.receive_command()
            if data is None:
                continue
            try:
                await self._dispatch_command(data.get("command", ""), data)
            except Exception as exc:
                logger.exception("SO101 setup command failed")
                await self._send_event("error", message=str(exc), error_code="command_error")

    async def _dispatch_command(self, command: str, data: dict[str, Any]) -> None:
        if command == "ping":
            await self._send_event("pong")
            return
        async with self._operation_lock:
            if self._stop_requested:
                return
            match command:
                case "start_motor_setup":
                    self._set_phase(SetupPhase.MOTOR_SETUP)
                    self.probe_result = None
                    await self._reset_calibration_flow()
                    await self._send_phase_status("Motor setup mode — connect motors one at a time.")
                case "motor_connected":
                    await self._handle_motor_setup(data.get("motor", ""))
                case "finish_motor_setup":
                    self._require_phase(SetupPhase.MOTOR_SETUP)
                    await self._run_motor_probe()
                case "enter_calibration":
                    await self._enter_calibration()
                case "start_homing":
                    await self._handle_start_homing()
                case "start_recording":
                    await self._handle_start_recording()
                case "stop_recording":
                    await self._handle_stop_recording()
                case "enter_verification":
                    await self._enter_verification()
                case "re_probe":
                    await self._run_voltage_check()
                    await self._run_motor_probe()
                case _:
                    raise ValueError(f"Unknown command: {command}")

    async def _send_phase_status(self, message: str) -> None:
        await self._send_event("status", state=self.state.value, phase=self.phase.value, message=message)

    async def _send_event(self, event: str, **kwargs: Any) -> None:
        await self.transport.send_json({"event": event, **kwargs})

    async def _cleanup(self) -> None:
        self._stop_requested = True
        self._generation += 1
        async with self._operation_lock:
            session = self.setup_session
            if session is not None:
                try:
                    await self._hardware_call(session.disconnect, cleanup=True)
                finally:
                    self.setup_session = None

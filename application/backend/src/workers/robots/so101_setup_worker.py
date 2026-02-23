"""SO101 Setup Worker — websocket-driven state machine for robot setup.

Guides the user through:
  1. Voltage check — verify power source matches leader/follower selection
  2. Motor probe — check all 6 motors are present with correct model numbers
  3. Motor setup — if motors are missing, guide per-motor ID assignment (reimplements
     lerobot's interactive setup_motors without input() calls)
  4. Calibration — homing offsets + range-of-motion recording (reimplements lerobot's
     interactive calibrate without input() calls)

All steps are driven by websocket commands from the frontend. The worker holds its own
FeetechMotorsBus connection (with handshake=False) and never touches the DB.
"""

import asyncio
from enum import StrEnum
from typing import Any

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from loguru import logger

from utils.serial_robot_tools import find_port_for_serial
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState

# ---------------------------------------------------------------------------
# Constants (shared with cli_robot_setup.py)
# ---------------------------------------------------------------------------

STS3215_MODEL_NUMBER = 777
VOLTAGE_THRESHOLD_RAW = 70  # 7.0V in register units (0.1V per unit)
VOLTAGE_UNIT = 0.1

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _build_motors() -> dict[str, Motor]:
    """Build the motor dict for an SO101 arm (shared leader/follower layout)."""
    body = MotorNormMode.RANGE_M100_100
    grip = MotorNormMode.RANGE_0_100
    return {
        "shoulder_pan": Motor(1, "sts3215", body),
        "shoulder_lift": Motor(2, "sts3215", body),
        "elbow_flex": Motor(3, "sts3215", body),
        "wrist_flex": Motor(4, "sts3215", body),
        "wrist_roll": Motor(5, "sts3215", body),
        "gripper": Motor(6, "sts3215", grip),
    }


# ---------------------------------------------------------------------------
# Setup phases
# ---------------------------------------------------------------------------


class SetupPhase(StrEnum):
    """Phases of the setup wizard state machine."""

    CONNECTING = "connecting"
    VOLTAGE_CHECK = "voltage_check"
    MOTOR_PROBE = "motor_probe"
    MOTOR_SETUP = "motor_setup"  # Only entered if motors are missing
    CALIBRATION_HOMING = "calibration_homing"
    CALIBRATION_RECORDING = "calibration_recording"
    CONFIGURE = "configure"
    COMPLETE = "complete"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class SO101SetupWorker(TransportWorker):
    """Websocket worker that drives the SO101 setup wizard.

    Protocol overview (client sends commands, server sends events):

    After connection the worker immediately runs voltage check + motor probe
    and sends the results. Then it waits for commands:

    Commands:
        {"command": "ping"}
        {"command": "start_motor_setup"}        — begin per-motor ID assignment
        {"command": "motor_connected", "motor": "shoulder_pan"}  — user connected single motor
        {"command": "start_homing"}             — user centered robot, apply homing offsets
        {"command": "start_recording"}          — begin recording range-of-motion
        {"command": "stop_recording"}           — finish recording, write calibration
        {"command": "start_positions_stream"}   — start streaming raw positions (~30Hz)
        {"command": "stop_positions_stream"}    — stop streaming raw positions
        {"command": "stream_positions"}         — start streaming normalized positions (~20Hz)
        {"command": "stop_stream"}              — stop streaming positions

    Events sent to client:
        {"event": "status", "state": ..., "phase": ..., "message": ...}
        {"event": "voltage_result", ...}
        {"event": "motor_probe_result", ...}
        {"event": "motor_setup_progress", ...}
        {"event": "homing_result", ...}
        {"event": "positions", ...}
        {"event": "state_was_updated", "state": {...}}  — normalized positions for 3D preview
        {"event": "calibration_result", ...}
        {"event": "error", "message": ...}
    """

    def __init__(
        self,
        transport: WorkerTransport,
        robot_type: str,
        serial_number: str,
    ) -> None:
        super().__init__(transport)
        self.robot_type = robot_type  # "SO101_Follower" or "SO101_Leader"
        self.serial_number = serial_number
        self.phase = SetupPhase.CONNECTING

        self.bus: FeetechMotorsBus | None = None
        self.motors = _build_motors()

        # Results accumulated during the flow
        self.voltage_result: dict[str, Any] | None = None
        self.probe_result: dict[str, Any] | None = None
        self.homing_offsets: dict[str, int] | None = None

        # Range recording state
        self._recording = False
        self._range_mins: dict[str, int] = {}
        self._range_maxes: dict[str, int] = {}

        # Raw position streaming state (for live joint table in calibration)
        self._positions_streaming = False
        self._positions_fps = 30

        # Normalized position streaming state (for 3D preview verification)
        self._streaming = False

        # Background tasks (prevent GC of fire-and-forget asyncio tasks)
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Helpers — bus / homing guard
    # ------------------------------------------------------------------

    def _require_bus(self) -> FeetechMotorsBus:
        """Return the motor bus, raising if not connected."""
        if self.bus is None:
            raise RuntimeError("Motor bus is not connected")
        return self.bus

    def _require_homing_offsets(self) -> dict[str, int]:
        """Return homing offsets, raising if not yet computed."""
        if self.homing_offsets is None:
            raise RuntimeError("Homing offsets have not been computed yet")
        return self.homing_offsets

    def _spawn_task(self, coro: Any) -> None:
        """Create a background task and prevent it from being garbage-collected."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main worker lifecycle."""
        try:
            await self.transport.connect()
            self.state = WorkerState.RUNNING

            # Phase 1: Connect to the bus
            await self._connect_bus()

            # Phase 2: Voltage check (automatic)
            await self._run_voltage_check()

            # Phase 3: Motor probe (automatic)
            await self._run_motor_probe()

            # Now wait for commands from the frontend
            await self._command_loop()

        except Exception as e:
            self.state = WorkerState.ERROR
            self.phase = SetupPhase.ERROR
            self.error_message = str(e)
            logger.exception(f"Setup worker error: {e}")
            await self._send_event("error", message=str(e))
        finally:
            await self._cleanup()
            await self.shutdown()

    # ------------------------------------------------------------------
    # Phase: Connect
    # ------------------------------------------------------------------

    async def _connect_bus(self) -> None:
        """Resolve serial number and open the motor bus."""
        self.phase = SetupPhase.CONNECTING
        await self._send_phase_status("Resolving serial port...")

        port = find_port_for_serial(self.serial_number)
        if not port:
            raise ConnectionError(f"No USB device found with serial number '{self.serial_number}'")

        logger.info(f"Setup worker: connecting to {port} (serial={self.serial_number})")

        self.bus = FeetechMotorsBus(port=port, motors=self.motors)
        await asyncio.to_thread(self.bus.connect, handshake=False)

        await self._send_phase_status(f"Connected to {port}")

    # ------------------------------------------------------------------
    # Phase: Voltage check
    # ------------------------------------------------------------------

    async def _run_voltage_check(self) -> None:
        """Read voltage from motors and check against expected power source."""
        self.phase = SetupPhase.VOLTAGE_CHECK
        await self._send_phase_status("Checking supply voltage...")

        bus = self._require_bus()

        readings = []
        for name, motor in bus.motors.items():
            raw: int | None = None
            try:
                raw = int(await asyncio.to_thread(bus.read, "Present_Voltage", name, normalize=False))
            except Exception:
                logger.debug(f"Failed to read voltage for motor '{name}'", exc_info=True)
            readings.append({"name": name, "motor_id": motor.id, "raw": raw})

        # Compute average
        raw_values = [r["raw"] for r in readings if r["raw"] is not None]
        avg_raw = sum(raw_values) / len(raw_values) if raw_values else None
        avg_voltage = avg_raw * VOLTAGE_UNIT if avg_raw is not None else None

        is_follower = self.robot_type == "SO101_Follower"
        if avg_raw is not None:
            voltage_ok = avg_raw >= VOLTAGE_THRESHOLD_RAW if is_follower else avg_raw < VOLTAGE_THRESHOLD_RAW
        else:
            voltage_ok = True  # Can't determine — don't block

        expected_source = "external power supply (>= 7V)" if is_follower else "USB only (< 7V)"

        self.voltage_result = {
            "event": "voltage_result",
            "readings": readings,
            "avg_voltage": avg_voltage,
            "voltage_ok": voltage_ok,
            "expected_source": expected_source,
            "robot_type": self.robot_type,
        }

        await self.transport.send_json(self.voltage_result)

    # ------------------------------------------------------------------
    # Phase: Motor probe
    # ------------------------------------------------------------------

    async def _run_motor_probe(self) -> None:
        """Ping each expected motor and report status."""
        self.phase = SetupPhase.MOTOR_PROBE
        await self._send_phase_status("Probing motors...")

        bus = self._require_bus()

        motors_found = []
        for name, motor in bus.motors.items():
            model_nb = await asyncio.to_thread(bus.ping, motor.id)
            found = model_nb is not None
            model_correct = model_nb == STS3215_MODEL_NUMBER if found else False
            motors_found.append(
                {
                    "name": name,
                    "motor_id": motor.id,
                    "found": found,
                    "model_number": model_nb,
                    "model_correct": model_correct,
                }
            )

        all_ok = all(m["found"] and m["model_correct"] for m in motors_found)
        found_count = sum(1 for m in motors_found if m["found"] and m["model_correct"])

        # Also check if already calibrated
        calibration_status = None
        if all_ok:
            calibration_status = await self._check_calibration()

        self.probe_result = {
            "event": "motor_probe_result",
            "motors": motors_found,
            "all_motors_ok": all_ok,
            "found_count": found_count,
            "total_count": len(self.motors),
            "calibration": calibration_status,
        }

        await self.transport.send_json(self.probe_result)

    async def _check_calibration(self) -> dict[str, Any]:
        """Check calibration state of all motors from EEPROM."""
        bus = self._require_bus()

        cal = await asyncio.to_thread(bus.read_calibration)
        motors_cal = {}
        all_calibrated = True
        for name, mc in cal.items():
            is_default = mc.homing_offset == 0 and mc.range_min == 0 and mc.range_max == 4095
            motors_cal[name] = {
                "homing_offset": mc.homing_offset,
                "range_min": mc.range_min,
                "range_max": mc.range_max,
                "is_calibrated": not is_default,
            }
            if is_default:
                all_calibrated = False

        return {
            "motors": motors_cal,
            "all_calibrated": all_calibrated,
        }

    # ------------------------------------------------------------------
    # Phase: Motor setup (per-motor ID assignment)
    # ------------------------------------------------------------------

    async def _handle_motor_setup(self, motor_name: str) -> None:
        """Set up a single motor — user has connected only this motor.

        Reimplements lerobot's setup_motor() without input() calls.
        The frontend tells us which motor the user connected via command.
        """
        bus = self._require_bus()

        await self._send_event(
            "motor_setup_progress",
            motor=motor_name,
            status="scanning",
            message=f"Scanning for motor '{motor_name}'...",
        )

        try:
            # Use the bus's setup_motor method which handles scanning + ID/baudrate assignment
            await asyncio.to_thread(bus.setup_motor, motor_name)

            await self._send_event(
                "motor_setup_progress",
                motor=motor_name,
                status="success",
                message=f"Motor '{motor_name}' configured as ID {bus.motors[motor_name].id}",
            )
        except Exception as e:
            logger.error(f"Motor setup failed for {motor_name}: {e}")
            await self._send_event(
                "motor_setup_progress",
                motor=motor_name,
                status="error",
                message=str(e),
            )

    # ------------------------------------------------------------------
    # Phase: Calibration — homing offsets
    # ------------------------------------------------------------------

    async def _handle_start_homing(self) -> None:
        """User has centered the robot — compute and write homing offsets.

        Reimplements lerobot's set_half_turn_homings().
        """
        self.phase = SetupPhase.CALIBRATION_HOMING
        await self._send_phase_status("Applying homing offsets...")

        bus = self._require_bus()

        # Disable torque and set operating mode
        await asyncio.to_thread(bus.disable_torque)
        for motor in bus.motors:
            await asyncio.to_thread(
                bus.write,
                "Operating_Mode",
                motor,
                0,  # Position mode
            )

        # Apply homing offsets — narrow from dict[NameOrID, Value] to dict[str, int]
        raw_offsets = await asyncio.to_thread(bus.set_half_turn_homings)
        self.homing_offsets = {str(k): int(v) for k, v in raw_offsets.items()}

        result = {
            "event": "homing_result",
            "offsets": {name: int(offset) for name, offset in self.homing_offsets.items()},
        }
        await self.transport.send_json(result)

    # ------------------------------------------------------------------
    # Phase: Calibration — range-of-motion recording
    # ------------------------------------------------------------------

    async def _handle_start_recording(self) -> None:
        """Start recording range-of-motion. Sends position updates until stopped."""
        self.phase = SetupPhase.CALIBRATION_RECORDING
        await self._send_phase_status("Recording range of motion...")

        bus = self._require_bus()

        # Read initial positions
        start_positions = await asyncio.to_thread(bus.sync_read, "Present_Position", list(bus.motors), normalize=False)

        self._range_mins = {m: int(v) for m, v in start_positions.items()}
        self._range_maxes = {m: int(v) for m, v in start_positions.items()}
        self._recording = True

        # Stream positions until _recording is set to False
        while self._recording and not self._stop_requested:
            positions = await asyncio.to_thread(bus.sync_read, "Present_Position", list(bus.motors), normalize=False)

            for motor, pos_val in positions.items():
                pos = int(pos_val)
                self._range_mins[motor] = min(self._range_mins[motor], pos)
                self._range_maxes[motor] = max(self._range_maxes[motor], pos)

            await self.transport.send_json(
                {
                    "event": "positions",
                    "motors": {
                        name: {
                            "position": int(positions[name]),
                            "min": self._range_mins[name],
                            "max": self._range_maxes[name],
                        }
                        for name in bus.motors
                    },
                }
            )

            await asyncio.sleep(0.05)  # ~20Hz

    async def _handle_stop_recording(self) -> None:
        """Stop recording and write calibration to motor EEPROM."""
        self._recording = False

        # Small delay to let the recording loop finish
        await asyncio.sleep(0.1)

        bus = self._require_bus()
        homing_offsets = self._require_homing_offsets()

        # Validate that min != max for all motors
        same_min_max = [m for m in bus.motors if self._range_mins.get(m, 0) == self._range_maxes.get(m, 0)]
        if same_min_max:
            await self._send_event(
                "error",
                message=f"Some motors have the same min and max values: {same_min_max}. "
                "Please move all joints through their full range.",
            )
            return

        # Build calibration dict and write to motor EEPROM
        calibration: dict[str, MotorCalibration] = {}
        for motor_name, motor_obj in bus.motors.items():
            calibration[motor_name] = MotorCalibration(
                id=motor_obj.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor_name],
                range_min=self._range_mins[motor_name],
                range_max=self._range_maxes[motor_name],
            )

        await asyncio.to_thread(bus.write_calibration, calibration)

        # Now configure the motors (return delay, acceleration, PID, etc.)
        await self._configure_motors()

        self.phase = SetupPhase.COMPLETE
        await self._send_phase_status("Calibration complete")

        # Send the final calibration data back
        await self.transport.send_json(
            {
                "event": "calibration_result",
                "calibration": {
                    name: {
                        "id": cal.id,
                        "drive_mode": cal.drive_mode,
                        "homing_offset": cal.homing_offset,
                        "range_min": cal.range_min,
                        "range_max": cal.range_max,
                    }
                    for name, cal in calibration.items()
                },
            }
        )

    # ------------------------------------------------------------------
    # Configure motors (PID, acceleration, operating mode)
    # ------------------------------------------------------------------

    async def _configure_motors(self) -> None:
        """Apply motor configuration — reimplements SO101Follower.configure()."""
        self.phase = SetupPhase.CONFIGURE
        await self._send_phase_status("Configuring motors...")

        bus = self._require_bus()

        # Disable torque for configuration
        await asyncio.to_thread(bus.disable_torque)

        # Configure bus-level settings (return delay, acceleration)
        await asyncio.to_thread(bus.configure_motors)

        # Per-motor PID and operating mode
        is_follower = self.robot_type == "SO101_Follower"
        for motor in bus.motors:
            await asyncio.to_thread(bus.write, "Operating_Mode", motor, 0)  # Position mode
            await asyncio.to_thread(bus.write, "P_Coefficient", motor, 16)
            await asyncio.to_thread(bus.write, "I_Coefficient", motor, 0)
            await asyncio.to_thread(bus.write, "D_Coefficient", motor, 32)

            if motor == "gripper":
                await asyncio.to_thread(bus.write, "Max_Torque_Limit", motor, 500)
                await asyncio.to_thread(bus.write, "Protection_Current", motor, 250)
                await asyncio.to_thread(bus.write, "Overload_Torque", motor, 25)

        # For follower: enable torque. For leader: leave disabled (moved by hand).
        if is_follower:
            await asyncio.to_thread(bus.enable_torque)

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------

    async def _command_loop(self) -> None:
        """Wait for and handle commands from the frontend."""
        while not self._stop_requested:
            data = await self.transport.receive_command()
            if data is None:
                continue

            command = data.get("command", "")
            logger.debug(f"Setup worker received command: {command}")

            try:
                await self._dispatch_command(command, data)
            except Exception as e:
                logger.exception(f"Error handling command '{command}': {e}")
                await self._send_event("error", message=str(e))

    async def _dispatch_command(self, command: str, data: dict[str, Any]) -> None:  # noqa: PLR0912
        """Dispatch a single command received from the frontend."""
        match command:
            case "ping":
                await self.transport.send_json({"event": "pong"})

            case "start_motor_setup":
                self.phase = SetupPhase.MOTOR_SETUP
                await self._send_phase_status("Motor setup mode — connect motors one at a time.")

            case "motor_connected":
                motor_name = data.get("motor", "")
                if motor_name not in self.motors:
                    await self._send_event("error", message=f"Unknown motor: {motor_name}")
                else:
                    await self._handle_motor_setup(motor_name)

            case "finish_motor_setup":
                # Re-run motor probe after setup
                await self._run_motor_probe()

            case "start_homing":
                await self._handle_start_homing()

            case "start_recording":
                # Run recording in a separate task so we can still receive commands
                self._spawn_task(self._handle_start_recording())

            case "stop_recording":
                await self._handle_stop_recording()

            case "start_positions_stream":
                fps = data.get("fps", self._positions_fps)
                self._positions_fps = max(1, min(60, int(fps)))
                self._spawn_task(self._handle_positions_stream())

            case "stop_positions_stream":
                self._positions_streaming = False

            case "stream_positions":
                # Run streaming in a separate task so we can still receive commands
                self._spawn_task(self._handle_stream_positions())

            case "stop_stream":
                await self._handle_stop_stream()

            case "re_probe":
                await self._run_voltage_check()
                await self._run_motor_probe()

            case _:
                await self._send_event("error", message=f"Unknown command: {command}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _handle_positions_stream(self) -> None:
        """Stream raw positions at the configured FPS until stopped.

        Sends 'positions' events only when values have changed since
        the last send, to avoid flooding the websocket when the robot
        is stationary.
        """
        bus = self._require_bus()

        # If already streaming, don't start a second loop
        if self._positions_streaming:
            return
        self._positions_streaming = True

        last_sent: dict[str, int] | None = None
        interval = 1.0 / self._positions_fps

        while self._positions_streaming and not self._stop_requested:
            try:
                positions = await asyncio.to_thread(
                    bus.sync_read, "Present_Position", list(bus.motors), normalize=False
                )
                current = {name: int(val) for name, val in positions.items()}

                # Only send when something changed
                if current != last_sent:
                    await self.transport.send_json(
                        {
                            "event": "positions",
                            "motors": {name: {"position": val} for name, val in current.items()},
                        }
                    )
                    last_sent = current

            except Exception as e:
                logger.warning(f"Position stream read error: {e}")

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Position streaming (for 3D preview in verification step)
    # ------------------------------------------------------------------

    async def _handle_stream_positions(self) -> None:
        """Start streaming normalized positions for 3D preview.

        Sends 'state_was_updated' events in the same format as the standard
        RobotWorker broadcast loop, so the frontend can reuse the same joint
        sync logic. Values are normalized (-100..100 for body, 0..100 for gripper)
        which the frontend treats as degrees and converts with degToRad().
        """
        bus = self._require_bus()

        # Ensure calibration is loaded — it may not be if the user skipped
        # straight to verification (robot was already calibrated from a prior
        # session). read_calibration() reads from motor EEPROM and returns
        # dict[str, MotorCalibration]. We assign it directly to bus.calibration
        # so _normalize() can use it — no need to write back to EEPROM.
        if not bus.calibration:
            logger.info("Loading calibration from motor EEPROM for position streaming")
            cal = await asyncio.to_thread(bus.read_calibration)
            bus.calibration = cal

            # Send calibration_result so the frontend has calibration data for
            # the save flow, even when the user skipped the calibration step.
            await self.transport.send_json(
                {
                    "event": "calibration_result",
                    "calibration": {
                        name: {
                            "id": mc.id,
                            "drive_mode": mc.drive_mode,
                            "homing_offset": mc.homing_offset,
                            "range_min": mc.range_min,
                            "range_max": mc.range_max,
                        }
                        for name, mc in cal.items()
                    },
                }
            )

        self._streaming = True

        while self._streaming and not self._stop_requested:
            try:
                state = await asyncio.to_thread(bus.sync_read, "Present_Position", list(bus.motors), normalize=True)

                await self.transport.send_json(
                    {
                        "event": "state_was_updated",
                        "state": {f"{name}.pos": float(val) for name, val in state.items()},
                    }
                )
            except Exception as e:
                logger.warning(f"Position streaming read error: {e}")

            await asyncio.sleep(0.05)  # ~20Hz

    async def _handle_stop_stream(self) -> None:
        """Stop streaming positions."""
        self._streaming = False

    async def _send_phase_status(self, message: str) -> None:
        """Send a status event with current phase info."""
        await self.transport.send_json(
            {
                "event": "status",
                "state": self.state.value,
                "phase": self.phase.value,
                "message": message,
            }
        )

    async def _send_event(self, event: str, **kwargs: Any) -> None:
        """Send a named event with arbitrary payload."""
        await self.transport.send_json({"event": event, **kwargs})

    async def _cleanup(self) -> None:
        """Disconnect the motor bus."""
        self._recording = False
        self._streaming = False
        if self.bus is not None:
            try:
                await asyncio.to_thread(self.bus.disconnect)
            except Exception:
                try:
                    self.bus.port_handler.closePort()
                except Exception:
                    logger.debug("Failed to close motor bus port", exc_info=True)
            self.bus = None

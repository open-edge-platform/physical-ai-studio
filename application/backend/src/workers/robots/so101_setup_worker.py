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
import time
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
FPS = 30  # Streaming rate for position reads (calibration + 3D preview)

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
    """Phases of the setup wizard state machine.

    The broadcast loop uses this to decide what to stream:
      - CALIBRATION_INSTRUCTIONS / CALIBRATION_HOMING → raw positions
        (``normalize=False``) with change detection (skip unchanged frames)
      - CALIBRATION_RECORDING → raw positions (``normalize=False``) AND
        track per-motor min/max for range-of-motion calibration
      - VERIFICATION → normalized positions (``normalize=True``)
      - Everything else → no streaming (idle sleep)

    During CALIBRATION_RECORDING the broadcast loop additionally tracks
    per-motor min/max values for range-of-motion calibration.
    """

    CONNECTING = "connecting"
    VOLTAGE_CHECK = "voltage_check"
    MOTOR_PROBE = "motor_probe"
    MOTOR_SETUP = "motor_setup"  # Only entered if motors are missing
    CALIBRATION_INSTRUCTIONS = "calibration_instructions"
    CALIBRATION_HOMING = "calibration_homing"
    CALIBRATION_RECORDING = "calibration_recording"
    CONFIGURE = "configure"
    VERIFICATION = "verification"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class SO101SetupWorker(TransportWorker):
    """Websocket worker that drives the SO101 setup wizard.

    Architecture mirrors ``RobotWorker``: two concurrent asyncio tasks —
    a *broadcast loop* that reads the motor bus and pushes events to the
    client, and a *command loop* that receives commands from the client.

    Streaming behaviour is controlled entirely by ``self.phase``:

      - ``CALIBRATION_INSTRUCTIONS`` / ``CALIBRATION_HOMING`` → stream
        raw positions (``normalize=False``) with change detection.
      - ``CALIBRATION_RECORDING`` → stream raw positions AND track
        per-motor min/max for range-of-motion calibration.
      - ``VERIFICATION`` → stream normalized positions
        (``normalize=True``), sending ``state_was_updated`` events that
        match the format used by the standard ``RobotWorker`` broadcast.
      - All other phases → no streaming (the loop idles).

    Commands:
        {"command": "ping"}
        {"command": "start_motor_setup"}
        {"command": "motor_connected", "motor": "shoulder_pan"}
        {"command": "finish_motor_setup"}
        {"command": "enter_calibration"}
        {"command": "start_homing"}
        {"command": "start_recording"}
        {"command": "stop_recording"}
        {"command": "enter_verification"}
        {"command": "re_probe"}

    Events sent to client:
        {"event": "status", "state": ..., "phase": ..., "message": ...}
        {"event": "voltage_result", ...}
        {"event": "motor_probe_result", ...}
        {"event": "motor_setup_progress", ...}
        {"event": "homing_result", ...}
        {"event": "positions", ...}
        {"event": "state_was_updated", "state": {...}}
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

        # Serialize all bus I/O — the Feetech SDK has no internal locking,
        # so concurrent asyncio.to_thread(bus.*) calls from the broadcast
        # loop and command handlers would collide on the serial port,
        # causing "[TxRxResult] Port is in use!" errors.
        self._bus_lock = asyncio.Lock()

        # Results accumulated during the flow
        self.voltage_result: dict[str, Any] | None = None
        self.probe_result: dict[str, Any] | None = None
        self.homing_offsets: dict[str, int] | None = None

        # Range recording state (used by broadcast loop during CALIBRATION_RECORDING)
        self._range_mins: dict[str, int] = {}
        self._range_maxes: dict[str, int] = {}

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

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main worker lifecycle.

        Connects to the motor bus, runs initial diagnostics (voltage +
        motor probe), then starts two concurrent tasks:
          - ``_broadcast_loop`` — reads the bus and pushes events
          - ``_command_loop`` — receives and dispatches frontend commands
        """
        try:
            await self.transport.connect()
            self.state = WorkerState.RUNNING

            # Phase 1: Connect to the bus
            await self._connect_bus()

            # Phase 2: Voltage check (automatic)
            await self._run_voltage_check()

            # Phase 3: Motor probe (automatic)
            await self._run_motor_probe()

            # Two concurrent tasks — mirrors RobotWorker architecture
            await self.run_concurrent(
                asyncio.create_task(self._broadcast_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
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
        async with self._bus_lock:
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
                async with self._bus_lock:
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
            async with self._bus_lock:
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

        async with self._bus_lock:
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
            async with self._bus_lock:
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
        async with self._bus_lock:
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
        """Start recording range-of-motion.

        Reads the current positions to initialise min/max, then
        transitions to ``CALIBRATION_RECORDING``.  The broadcast loop
        picks up from here — it streams positions AND tracks min/max.
        """
        if self.phase == SetupPhase.CALIBRATION_RECORDING:
            return
        bus = self._require_bus()

        await self._send_phase_status("Recording range of motion...")

        # Read initial positions to seed min/max
        async with self._bus_lock:
            start_positions = await asyncio.to_thread(
                bus.sync_read, "Present_Position", list(bus.motors), normalize=False
            )

        self._range_mins = {m: int(v) for m, v in start_positions.items()}
        self._range_maxes = {m: int(v) for m, v in start_positions.items()}

        # The broadcast loop will start tracking min/max on next iteration
        self.phase = SetupPhase.CALIBRATION_RECORDING

    async def _handle_stop_recording(self) -> None:
        """Stop recording and write calibration to motor EEPROM.

        Transitions the phase to ``CONFIGURE`` so the broadcast loop
        stops streaming.  Because the command handler and broadcast loop
        share ``_bus_lock``, the lock acquisition here guarantees no bus
        reads are in flight when we start writing calibration.
        """
        # Transition phase — broadcast loop will idle on next iteration
        self.phase = SetupPhase.CONFIGURE

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

        async with self._bus_lock:
            await asyncio.to_thread(bus.write_calibration, calibration)

        # Now configure the motors (return delay, acceleration, PID, etc.)
        await self._configure_motors()

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

        # Transition to verification — broadcast loop auto-starts
        # normalized streaming on the next iteration.
        await self._enter_verification()

    async def _enter_verification(self) -> None:
        """Enter VERIFICATION phase, ensuring calibration is loaded on the bus.

        If the user went through the full calibration flow, the bus
        already has ``bus.calibration`` populated from ``write_calibration``.
        If the user skipped calibration (robot was already calibrated from
        a prior session), we load it from motor EEPROM.
        """
        bus = self._require_bus()

        if not bus.calibration:
            logger.info("Loading calibration from motor EEPROM for verification streaming")
            async with self._bus_lock:
                cal = await asyncio.to_thread(bus.read_calibration)
            bus.calibration = cal

            # Send calibration_result so the frontend has calibration data
            # for the save flow, even when calibration was skipped.
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

        self.phase = SetupPhase.VERIFICATION

    # ------------------------------------------------------------------
    # Configure motors (PID, acceleration, operating mode)
    # ------------------------------------------------------------------

    async def _configure_motors(self) -> None:
        """Apply motor configuration — reimplements SO101Follower.configure()."""
        await self._send_phase_status("Configuring motors...")

        bus = self._require_bus()

        async with self._bus_lock:
            # Disable torque for configuration
            await asyncio.to_thread(bus.disable_torque)

            # Configure bus-level settings (return delay, acceleration)
            await asyncio.to_thread(bus.configure_motors)

            # Per-motor PID and operating mode
            is_follower = self.robot_type == "SO101_Follower"
            for motor in bus.motors:
                await asyncio.to_thread(bus.write, "Operating_Mode", motor, 0)  # Position mode
                if is_follower:
                    await asyncio.to_thread(bus.write, "P_Coefficient", motor, 16)
                    await asyncio.to_thread(bus.write, "I_Coefficient", motor, 0)
                    await asyncio.to_thread(bus.write, "D_Coefficient", motor, 32)

                    if motor == "gripper":
                        await asyncio.to_thread(bus.write, "Max_Torque_Limit", motor, 500)
                        await asyncio.to_thread(bus.write, "Protection_Current", motor, 250)
                        await asyncio.to_thread(bus.write, "Overload_Torque", motor, 25)

    # ------------------------------------------------------------------
    # Broadcast loop — phase-driven streaming
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Read the motor bus and push events to the client.

        Behaviour depends on the current phase:

          - ``CALIBRATION_INSTRUCTIONS`` / ``CALIBRATION_HOMING`` — stream
            raw positions so the user can see live joint values (while
            centering or while instructions are displayed).
          - ``CALIBRATION_RECORDING`` — stream raw positions AND track
            per-motor min/max for range-of-motion calibration.
          - ``VERIFICATION`` — stream normalized positions using the
            same ``state_was_updated`` event format as ``RobotWorker``.
          - All other phases — idle (no bus reads).
        """
        read_interval = 1.0 / FPS

        # Tracks last-sent raw positions so we can skip unchanged frames
        # during CALIBRATION_HOMING.  Mutable list-of-one so
        # _broadcast_raw_positions can update it.
        last_raw: list[dict[str, int] | None] = [None]

        try:
            while not self._stop_requested:
                start_time = time.perf_counter()

                try:
                    await self._broadcast_tick(last_raw)
                except Exception as e:
                    logger.warning(f"Broadcast loop error: {e}")

                elapsed = time.perf_counter() - start_time
                await asyncio.sleep(max(0.001, read_interval - elapsed))
        except asyncio.CancelledError:
            pass

    async def _broadcast_tick(self, last_raw: list[dict[str, int] | None]) -> None:
        """Single iteration of the broadcast loop, dispatched by phase."""
        match self.phase:
            case SetupPhase.CALIBRATION_INSTRUCTIONS | SetupPhase.CALIBRATION_HOMING:
                await self._broadcast_raw_positions(last_raw, track_range=False)

            case SetupPhase.CALIBRATION_RECORDING:
                await self._broadcast_raw_positions(last_raw, track_range=True)

            case SetupPhase.VERIFICATION:
                await self._broadcast_normalized_positions()

            case _:
                # No streaming needed — the sleep in _broadcast_loop
                # keeps us from busy-spinning.
                pass

    async def _broadcast_raw_positions(self, last_raw: list[dict[str, int] | None], *, track_range: bool) -> None:
        """Read raw positions and send a ``positions`` event.

        When *track_range* is True (``CALIBRATION_RECORDING``), also
        updates ``_range_mins`` / ``_range_maxes`` and includes min/max
        in the event payload.

        *last_raw* is a mutable single-element list so the caller's
        reference is updated in place (used for change-detection).
        """
        bus = self._require_bus()

        async with self._bus_lock:
            positions = await asyncio.to_thread(bus.sync_read, "Present_Position", list(bus.motors), normalize=False)

        current = {name: int(val) for name, val in positions.items()}

        if track_range:
            for motor, pos in current.items():
                self._range_mins[motor] = min(self._range_mins[motor], pos)
                self._range_maxes[motor] = max(self._range_maxes[motor], pos)

            await self.transport.send_json(
                {
                    "event": "positions",
                    "motors": {
                        name: {
                            "position": pos,
                            "min": self._range_mins[name],
                            "max": self._range_maxes[name],
                        }
                        for name, pos in current.items()
                    },
                }
            )
        elif current != last_raw[0]:
            # Only send when values changed (avoid flooding during idle centering)
            await self.transport.send_json(
                {
                    "event": "positions",
                    "motors": {name: {"position": val} for name, val in current.items()},
                }
            )

        last_raw[0] = current

    async def _broadcast_normalized_positions(self) -> None:
        """Read normalized positions and send a ``state_was_updated`` event.

        Uses the same event format as ``RobotWorker._broadcast_loop`` so
        the frontend can reuse its joint-sync logic.
        """
        bus = self._require_bus()

        async with self._bus_lock:
            state = await asyncio.to_thread(bus.sync_read, "Present_Position", list(bus.motors), normalize=True)

        await self.transport.send_json(
            {
                "event": "state_was_updated",
                "state": {f"{name}.pos": float(val) for name, val in state.items()},
            }
        )

    # ------------------------------------------------------------------
    # Command loop
    # ------------------------------------------------------------------

    async def _command_loop(self) -> None:
        """Wait for and handle commands from the frontend."""
        try:
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
        except asyncio.CancelledError:
            pass

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

            case "enter_calibration":
                self.phase = SetupPhase.CALIBRATION_INSTRUCTIONS

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
                await self._send_event("error", message=f"Unknown command: {command}")

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

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
        if self.bus is not None:
            try:
                async with self._bus_lock:
                    await asyncio.to_thread(self.bus.disconnect)
            except Exception:
                try:
                    self.bus.port_handler.closePort()
                except Exception:
                    logger.debug("Failed to close motor bus port", exc_info=True)
            self.bus = None

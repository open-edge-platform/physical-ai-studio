"""Robot Calibration Worker — websocket-driven zero-pose calibration for plugin robots.

Runs the ``RobotZeroCalibration`` a catalog definition provides. Guides the user through:
  1. Connect — build the plain driver from the unsaved robot and connect to it directly
  2. Positioning — release the arm and stream live joint positions while the user
     moves it into the zero pose described by the plugin's instructions
  3. Verification — after ``set_zero``, check every joint reads within the plugin's
     tolerance of zero; the user can retry ``set_zero`` until it passes

The worker holds its own exclusive driver connection (not a ``SharedRobot``, which only
forwards observations and actions) and never touches the DB: the UI creates the robot
once calibration succeeds, as it does after the SO101 setup wizard.
"""

import asyncio
import math
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import anyio
from loguru import logger
from pydantic import ValidationError

from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_client_factory import RobotClientFactory
from runtime.features import observation_to_dict
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState

if TYPE_CHECKING:
    from physicalai_studio_plugin import RobotZeroCalibration

FPS = 30  # Streaming rate for the live 3D view


class ZeroCalibrationUnsupportedError(ValueError):
    """The robot type's catalog definition offers no calibration."""


class ZeroCalibrationPhase(StrEnum):
    """Phases of the calibration state machine.

    The broadcast loop streams observations in ``POSITIONING`` and ``VERIFICATION``
    and idles otherwise.
    """

    WAITING = "waiting"
    CONNECTING = "connecting"
    POSITIONING = "positioning"
    VERIFICATION = "verification"


class RobotZeroCalibrationWorker(TransportWorker):
    """Websocket worker that runs a plugin's zero-pose calibration.

    Commands:
        {"command": "ping"}
        {"command": "start", "robot": {"id": ..., "name": ..., "type": ..., "payload": {...}}}
        {"command": "set_zero"}

    Events sent to client:
        {"event": "status", "state": ..., "phase": ..., "message": ...}
        {"event": "observation", "data": {"<joint>.pos": ...}}
        {"event": "calibration_result", "success": ..., "joints": {...}, "tolerance_deg": ...}
        {"event": "error", "message": ..., "error_code": ...}
    """

    def __init__(
        self,
        transport: WorkerTransport,
        robot_client_factory: RobotClientFactory,
        catalog_registry: RobotCatalogRegistry,
    ) -> None:
        super().__init__(transport)
        self.robot_client_factory = robot_client_factory
        self.catalog_registry = catalog_registry
        self.phase = ZeroCalibrationPhase.WAITING

        self.driver: Any | None = None
        self.zero_calibration: RobotZeroCalibration | None = None
        # Drivers are not thread-safe: serialize the broadcast loop's reads with set_zero.
        self._driver_lock = asyncio.Lock()

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Map an exception to a frontend-friendly error code."""
        if isinstance(exc, ZeroCalibrationUnsupportedError):
            return "zero_calibration_unsupported"
        if isinstance(exc, ValidationError):
            return "invalid_config"
        if isinstance(exc, PermissionError):
            return "permission_denied"
        if isinstance(exc, ConnectionError):
            return "device_not_found"
        return "connection_failed"

    def _require_driver(self) -> Any:
        """Return the connected driver, raising if calibration has not started."""
        if self.driver is None or self.zero_calibration is None:
            raise RuntimeError("Calibration has not started")
        return self.driver

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main worker lifecycle: a broadcast loop and a command loop, as in the SO101 setup worker."""
        try:
            await self.transport.connect()
            self.state = WorkerState.RUNNING
            await self._send_phase_status("Waiting for the robot to calibrate.")

            await self.run_concurrent(
                asyncio.create_task(self._broadcast_loop()),
                asyncio.create_task(self._command_loop()),
            )
        finally:
            await self._cleanup()
            await self.shutdown()

    # ------------------------------------------------------------------
    # Phase: Connect
    # ------------------------------------------------------------------

    async def _start(self, robot_data: Any) -> None:
        """Validate the unsaved robot, connect its driver directly, and release the arm."""
        if self.driver is not None:
            raise RuntimeError("Calibration has already started")

        self.phase = ZeroCalibrationPhase.CONNECTING
        await self._send_phase_status("Connecting to the robot...")

        robot = self.catalog_registry.get_robot_adapter().validate_python(robot_data)
        definition = self.catalog_registry.get_definition(robot.type)
        calibration = None if definition is None else definition.zero_calibration
        if calibration is None:
            raise ZeroCalibrationUnsupportedError(f"Robot type {robot.type} does not support calibration")

        driver, _definition = await self.robot_client_factory.build_robot_driver(robot, self.robot_client_factory)
        async with self._driver_lock:
            await asyncio.to_thread(driver.connect)
            self.driver = driver
            self.zero_calibration = calibration
            if calibration.release is not None:
                await calibration.release(driver)

        logger.info(f"Calibration worker: connected to {robot.type} robot {robot.name!r}")
        self.phase = ZeroCalibrationPhase.POSITIONING
        await self._send_phase_status(calibration.instructions)

    # ------------------------------------------------------------------
    # Phase: Verification
    # ------------------------------------------------------------------

    async def _set_zero(self) -> None:
        """Run the plugin's set-zero step, then check every joint reads close to zero."""
        driver = self._require_driver()
        calibration = self.zero_calibration
        if calibration is None:
            raise RuntimeError("Calibration has not started")

        async with self._driver_lock:
            await calibration.set_zero(driver)
            observation = await asyncio.to_thread(driver.get_observation)

        joints = observation_to_dict(driver.joint_names, observation, include_velocities=False)
        tolerance = calibration.zero_tolerance_deg
        success = all(math.isfinite(value) and abs(value) <= tolerance for value in joints.values())
        logger.info(f"Calibration worker: set zero, success={success}")

        self.phase = ZeroCalibrationPhase.VERIFICATION
        await self._send_event("calibration_result", success=success, joints=joints, tolerance_deg=tolerance)

    # ------------------------------------------------------------------
    # Broadcast loop
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Stream joint observations while the user positions the arm or checks the result."""
        read_interval = 1.0 / FPS

        try:
            while not self._stop_requested:
                start_time = time.perf_counter()

                if self.phase in {ZeroCalibrationPhase.POSITIONING, ZeroCalibrationPhase.VERIFICATION}:
                    try:
                        await self._broadcast_observation()
                    except Exception as e:
                        logger.warning(f"Broadcast loop error: {e}")

                elapsed = time.perf_counter() - start_time
                await asyncio.sleep(max(0.001, read_interval - elapsed))
        except asyncio.CancelledError:
            pass

    async def _broadcast_observation(self) -> None:
        """Read the driver and send an ``observation`` event, in the robot observation stream's format."""
        driver = self._require_driver()
        async with self._driver_lock:
            observation = await asyncio.to_thread(driver.get_observation)
        data = observation_to_dict(driver.joint_names, observation, include_velocities=False)
        await self._send_event("observation", data=data)

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
                logger.debug(f"Calibration worker received command: {command}")

                try:
                    await self._dispatch_command(command, data)
                except Exception as e:
                    logger.exception(f"Error handling command '{command}': {e}")
                    error_code = (
                        RobotZeroCalibrationWorker._classify_error(e) if command == "start" else "command_error"
                    )
                    await self._send_event("error", message=str(e), error_code=error_code)
        except asyncio.CancelledError:
            pass

    async def _dispatch_command(self, command: str, data: dict[str, Any]) -> None:
        """Dispatch a single command received from the frontend."""
        match command:
            case "ping":
                await self.transport.send_json({"event": "pong"})

            case "start":
                try:
                    await self._start(data.get("robot"))
                except Exception:
                    # A failed release leaves the driver connected: drop it so the user can retry.
                    await self._cleanup()
                    self.zero_calibration = None
                    self.phase = ZeroCalibrationPhase.WAITING
                    raise

            case "set_zero":
                await self._set_zero()

            case _:
                await self._send_event("error", message=f"Unknown command: {command}", error_code="command_error")

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
        """Disconnect the driver."""
        driver = self.driver
        self.driver = None
        if driver is None:
            return
        # Shield so a cancelled websocket task still releases the hardware. Starlette
        # cancels the ASGI task on client close; an unshielded await here would skip
        # disconnect and leave the arm's bus open (see api/robot_observations.py).
        with anyio.CancelScope(shield=True):
            try:
                async with self._driver_lock:
                    await asyncio.to_thread(driver.disconnect)
            except Exception:
                logger.warning("Failed to disconnect robot after calibration", exc_info=True)

from robots.utils import get_robot_client
import asyncio
import time

from loguru import logger

from schemas.robot import Robot
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.robot import RobotConnectionManager
from workers.robots.commands import handle_command, parse_command
from robots.robot_client import RobotClient
from workers.transport.worker_transport import WorkerTransport
from workers.transport_worker import TransportWorker, WorkerState, WorkerStatus


class RobotWorker(TransportWorker):
    """Orchestrates robot communication over configurable transport."""

    def __init__(
        self,
        robot: Robot,
        transport: WorkerTransport,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
        fps: int = 30,
        normalize: bool = False,
    ) -> None:
        super().__init__(transport)
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service

        self.robot = robot
        self.client: RobotClient | None = None

        self.fps = fps
        self.normalize = normalize

    async def run(self) -> None:
        """Main worker loop."""
        try:
            await self.transport.connect()

            self.client = await get_robot_client(self.robot, self.robot_manager, self.calibration_service)

            try:
                await self.client.connect()
                self.state = WorkerState.RUNNING

                logger.info(f"Created new robot client connection: {self.robot.id}")
                await self.transport.send_json(WorkerStatus(state=self.state, message="Robot connected").to_json())
            except Exception as e:
                logger.error(f"Failed to connect robot client: {e}")
                raise

            await self.run_concurrent(
                asyncio.create_task(self._broadcast_loop()),
                asyncio.create_task(self._command_loop()),
            )

        except Exception as e:
            self.state = WorkerState.ERROR
            self.error_message = str(e)
            logger.error(f"Worker error: {e}")
            await self.transport.send_json(WorkerStatus(state=self.state, message=str(e)).to_json())
        finally:
            await self.shutdown()

    async def _broadcast_loop(self) -> None:
        """Listen to robot state updates and forward to client."""
        read_interval = 1 / self.fps
        try:
            previous_values = None

            while not self._stop_requested:
                if self.client is None:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.perf_counter()
                try:
                    state = await self.client.read_state(normalize=self.normalize)

                    # Only send if joint values changed (ignore timestamp)
                    current_values = state.get("state")
                    if current_values != previous_values:
                        previous_values = current_values
                        self.last_state = state
                        await self.transport.send_json(state)

                except Exception as e:
                    logger.error(f"Error reading robot state: {e}")
                    await asyncio.sleep(1)

                elapsed = time.perf_counter() - start_time
                sleep_time = max(0.001, read_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass

    async def _command_loop(self) -> None:
        """Handle incoming commands from client."""
        try:
            while not self._stop_requested:
                command = await self.transport.receive_command()

                if not self.client or command is None:
                    continue

                try:
                    robot_command = parse_command(command)
                    response = await handle_command(self.client, robot_command)

                    if response:
                        await self.transport.send_json(response)

                except Exception as e:
                    logger.warning("Received unknown command: {} from command {}", e, command)
                    await self.transport.send_json(RobotClient._create_event("error", message=str(e)))
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info(f"Shutting down robot worker: {self.robot.id}")
        await super().shutdown()



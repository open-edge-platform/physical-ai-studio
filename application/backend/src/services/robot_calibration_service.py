import json
from uuid import UUID

import aiofiles
from lerobot.motors import MotorCalibration
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from loguru import logger

from db.engine import get_async_db_session_ctx
from exceptions import ResourceNotFoundError, ResourceType
from repositories.robot_calibration_repo import RobotCalibrationRepository
from schemas.calibration import Calibration
from schemas.robot import Robot, RobotType
from settings import Settings
from utils.robot import RobotConnectionManager


async def find_robot_port(manager: RobotConnectionManager, robot: Robot) -> str | None:
    """Find the port associated with a robot."""
    for managed_robot in manager.robots:
        if managed_robot.serial_id == robot.serial_id:
            return managed_robot.port

    return None


class RobotCalibrationService:
    robot_manager: RobotConnectionManager
    settings: Settings

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        settings: Settings,
    ):
        self.robot_manager = robot_manager
        self.settings = settings

    async def _save_calibration_to_disk(self, robot_id: UUID, calibration_data: Calibration) -> str:
        """Saves calibration data to a JSON file on disk."""
        calibration_dir = self.settings.robots_dir / str(robot_id) / "calibrations"
        calibration_dir.mkdir(parents=True, exist_ok=True)

        file_path = calibration_dir / f"{calibration_data.id}.json"

        # Prepare data for JSON serialization as per requirement
        json_data = {}
        for val in calibration_data.values.values():
            json_data[val.joint_name] = {
                "id": val.id,
                "drive_mode": val.drive_mode,
                "homing_offset": val.homing_offset,
                "range_min": val.range_min,
                "range_max": val.range_max,
            }

        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(json_data, indent=4))

        return str(file_path)

    async def save_calibration(self, robot_id: UUID, calibration_data: Calibration) -> Calibration:
        """
        Persists the calibration data for a specific robot.
        """
        logger.info(f"Saving calibration for robot_id: {robot_id}")

        # Ensure the calibration data corresponds to the robot_id provided
        calibration_data.robot_id = robot_id

        # Write calibration to disk
        calibration_data.file_path = await self._save_calibration_to_disk(robot_id, calibration_data)

        # Use repository to save the calibration data
        async with get_async_db_session_ctx() as db:
            repo = RobotCalibrationRepository(db)
            return await repo.save(calibration_data)

    async def get_robot_calibration(self, robot: Robot) -> list[Calibration]:
        async with get_async_db_session_ctx() as db:
            repo = RobotCalibrationRepository(db)
            return await repo.get_robot_calibration(robot.id)

    async def get_calibration(self, calibration_id: UUID) -> Calibration:
        async with get_async_db_session_ctx() as db:
            repo = RobotCalibrationRepository(db)
            calibration = await repo.get_by_id(calibration_id)

            if calibration is None:
                raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, str(calibration_id))

            return calibration

    async def get_robot_motor_calibration(self, robot: Robot) -> dict[str, MotorCalibration]:
        if robot.type not in (RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER):
            raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

        port = await find_robot_port(self.robot_manager, robot)

        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_id)

        # TODO: make this depend on the robot type
        # Assume follower since leader shares same FeetechMotorBus layout
        robot_config = SO101Follower(SO101FollowerConfig(port=port))
        robot_config.bus.connect()
        calibration = robot_config.bus.read_calibration()
        robot_config.bus.disconnect()

        return calibration

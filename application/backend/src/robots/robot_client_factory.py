from pathlib import Path

from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig

from exceptions import ResourceNotFoundError, ResourceType
from robots.robot_client import RobotClient
from robots.so101.so101_follower import SO101Follower
from robots.so101.so101_leader import SO101Leader
from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from robots.widowxai.trossen_widowx_ai_leader import TrossenWidowXAILeader
from schemas.calibration import Calibration
from schemas.robot import NetworkIpRobotConfig, Robot, RobotType
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.serial_robot_tools import RobotConnectionManager


class RobotClientFactory:
    calibration_service: RobotCalibrationService
    robot_manager: RobotConnectionManager

    def __init__(
        self,
        robot_manager: RobotConnectionManager,
        calibration_service: RobotCalibrationService,
    ) -> None:
        self.robot_manager = robot_manager
        self.calibration_service = calibration_service

    async def build(self, robot: Robot) -> RobotClient:
        match robot.type:
            case RobotType.TROSSEN_WIDOWXAI_FOLLOWER:
                config = NetworkIpRobotConfig(
                    type="follower",
                    robot_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
                    connection_string=robot.connection_string,
                )
                return TrossenWidowXAIFollower(config=config)
            case RobotType.TROSSEN_WIDOWXAI_LEADER:
                config = NetworkIpRobotConfig(
                    type="leader",
                    robot_type=RobotType.TROSSEN_WIDOWXAI_LEADER,
                    connection_string=robot.connection_string,
                )
                return TrossenWidowXAILeader(config=config)
            case RobotType.SO101_FOLLOWER:
                config = await self._get_robot_follower_config(robot)
                return SO101Follower(config)
            case RobotType.SO101_LEADER:
                config = await self.get_robot_leader_config(robot)
                return SO101Leader(config)

    async def _get_robot_follower_config(self, robot: Robot) -> SO101FollowerConfig:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            return SO101FollowerConfig(port=port)

        return SO101FollowerConfig(
            port=port, id=str(calibration.id), calibration_dir=Path(calibration.file_path).parent
        )

    async def get_robot_leader_config(self, robot: Robot) -> SO101LeaderConfig:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            return SO101LeaderConfig(port=port)

        return SO101LeaderConfig(port=port, id=str(calibration.id), calibration_dir=Path(calibration.file_path).parent)

    async def _find_robot_port(self, robot: Robot) -> str:
        port = await find_robot_port(self.robot_manager, robot)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)

        return port

    async def _get_robot_calibration(self, robot: Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None

        return await self.calibration_service.get_calibration(robot.active_calibration_id)

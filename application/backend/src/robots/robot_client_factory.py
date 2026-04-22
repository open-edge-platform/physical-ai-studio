from typing import Literal

from physicalai.robot.so101 import SO101, SO101Calibration
from physicalai.robot.trossen import WidowXAI

from exceptions import ResourceNotFoundError, ResourceType
from robots.robot_client import RobotClient
from robots.so101.adapter import SO101Adapter
from robots.widowxai.adapter import WidowXAIAdapter
from robots.widowxai.bimanual_adapter import BimanualWidowXAIAdapter
from schemas.calibration import Calibration
from schemas.robot import Robot, RobotType, SO101Robot, TrossenBimanualRobot
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
                trossen_robot = robot
                robot_driver = WidowXAI(ip=trossen_robot.payload.connection_string, role="follower")
                return WidowXAIAdapter(robot=robot_driver, mode="follower")
            case RobotType.TROSSEN_WIDOWXAI_LEADER:
                trossen_robot = robot
                robot_driver = WidowXAI(ip=trossen_robot.payload.connection_string, role="leader")
                return WidowXAIAdapter(robot=robot_driver, mode="leader")
            case RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER:
                return self._build_bimanual_widowxai(robot, mode="follower")
            case RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER:
                return self._build_bimanual_widowxai(robot, mode="leader")
            case RobotType.SO101_FOLLOWER:
                return await self._build_so101(robot)
            case RobotType.SO101_LEADER:
                return await self._build_so101(robot)
            case _:
                raise ValueError(f"Unsupported robot type: {robot.type}")

    @staticmethod
    def _build_bimanual_widowxai(
        robot: TrossenBimanualRobot, mode: Literal["follower", "leader"]
    ) -> BimanualWidowXAIAdapter:
        left_driver = WidowXAI(ip=robot.payload.connection_string_left, role=mode)
        right_driver = WidowXAI(ip=robot.payload.connection_string_right, role=mode)
        left_adapter = WidowXAIAdapter(robot=left_driver, mode=mode)
        right_adapter = WidowXAIAdapter(robot=right_driver, mode=mode)
        return BimanualWidowXAIAdapter(left=left_adapter, right=right_adapter, mode=mode)

    async def _build_so101(self, robot: SO101Robot) -> SO101Adapter:
        port = await self._find_robot_port(robot)
        calibration = await self._get_robot_calibration(robot)

        if calibration is None:
            raise ResourceNotFoundError(ResourceType.ROBOT_CALIBRATION, robot.payload.serial_number)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number)

        mode = "follower" if robot.type == RobotType.SO101_FOLLOWER else "teleoperator"
        role = "follower" if mode == "follower" else "leader"

        so101_cal = SO101Calibration.from_dict(
            {
                name: {
                    "id": val.id,
                    "drive_mode": val.drive_mode,
                    "homing_offset": val.homing_offset,
                    "range_min": val.range_min,
                    "range_max": val.range_max,
                }
                for name, val in calibration.values.items()
            }
        )

        so101 = SO101(port=port, calibration=so101_cal, role=role)
        return SO101Adapter(robot=so101, mode=mode, calibration=calibration)

    async def _find_robot_port(self, robot: SO101Robot) -> str:
        port = await find_robot_port(self.robot_manager, robot)
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.payload.serial_number)

        return port

    async def _get_robot_calibration(self, robot: SO101Robot) -> Calibration | None:
        if robot.active_calibration_id is None:
            return None

        return await self.calibration_service.get_calibration(robot.active_calibration_id)

import json

from lerobot.motors import MotorCalibration
from loguru import logger

from exceptions import ResourceNotFoundError, ResourceType
from robots.robot_client import RobotClient
from robots.so101 import So101
from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from robots.widowxai.trossen_widowx_ai_leader import TrossenWidowXAILeader
from schemas.robot import NetworkIpRobotConfig, Robot, RobotType
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.calibration import get_calibrations
from utils.serial_robot_tools import RobotConnectionManager


def _load_so101_calibration(robot: Robot) -> dict[str, MotorCalibration]:
    """Temporary utility function for getting calibration for So101.

    TODO: This should be done via calibration service once calibration arrives.
    """
    calibrations = get_calibrations()
    calibration_config = next(calibration for calibration in calibrations if calibration.id == robot.name.lower())

    with open(calibration_config.path) as f:
        return {key: MotorCalibration(**data) for key, data in json.load(f).items()}


async def get_robot_client(
    robot: Robot, robot_manager: RobotConnectionManager, _calibration_service: RobotCalibrationService
) -> RobotClient:
    """
    Get RobotClient based on robot-type.

    Still requires calibration implementation.
    Maybe move the specifics of instantiation to static function on respective RobotClient.
    """

    if robot.type in (RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER):
        calibration = _load_so101_calibration(robot)

        port = await find_robot_port(robot_manager, robot)
        logger.info(f"SO101: port: {port} id: {robot.name} {robot.type}")
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)
        mode = "follower" if robot.type == RobotType.SO101_FOLLOWER else "teleoperator"
        return So101(port=port, id=robot.name.lower(), mode=mode, calibration=calibration)
    if robot.type == RobotType.TROSSEN_WIDOWXAI_LEADER:
        config = NetworkIpRobotConfig(
            type="leader", robot_type=RobotType.TROSSEN_WIDOWXAI_LEADER, connection_string=robot.connection_string
        )
        return TrossenWidowXAILeader(config=config)
    if robot.type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER:
        config = NetworkIpRobotConfig(
            type="follower", robot_type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER, connection_string=robot.connection_string
        )
        return TrossenWidowXAIFollower(config=config)

    raise ValueError(f"No implementation for {robot.type}")

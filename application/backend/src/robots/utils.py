from loguru import logger

from exceptions import ResourceNotFoundError, ResourceType
from robots.robot_client import RobotClient
from robots.so101.so101_follower import SO101Follower
from robots.so101.so101_leader import SO101Leader
from robots.widowxai.trossen_widowx_ai_follower import TrossenWidowXAIFollower
from robots.widowxai.trossen_widowx_ai_leader import TrossenWidowXAILeader
from schemas.robot import NetworkIpRobotConfig, Robot, RobotType
from services.robot_calibration_service import RobotCalibrationService, find_robot_port
from utils.serial_robot_tools import RobotConnectionManager


async def get_robot_client(
    robot: Robot, robot_manager: RobotConnectionManager, _calibration_service: RobotCalibrationService
) -> RobotClient:
    """
    Get RobotClient based on robot-type.

    Still requires calibration implementation.
    Maybe move the specifics of instantiation to static function on respective RobotClient.
    """
    if robot.type == RobotType.SO101_FOLLOWER:
        port = await find_robot_port(robot_manager, robot)
        logger.info(f"Follower: port: {port} id: {robot.name}")
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)
        return SO101Follower(port=port, id=robot.name.lower())
    if robot.type == RobotType.SO101_LEADER:
        port = await find_robot_port(robot_manager, robot)
        logger.info(f"Leader: port: {port} id: {robot.name}")
        if port is None:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot.serial_number)
        return SO101Leader(port=port, id=robot.name.lower())
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

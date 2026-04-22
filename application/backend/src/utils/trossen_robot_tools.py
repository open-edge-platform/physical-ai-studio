import trossen_arm
from loguru import logger

from schemas import Robot
from schemas.robot import RobotType, TrossenBimanualRobot, TrossenSingleArmRobot

_TROSSEN_SINGLE_TYPES = {
    RobotType.TROSSEN_WIDOWXAI_LEADER,
    RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
}

_TROSSEN_BIMANUAL_TYPES = {
    RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER,
    RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER,
}


async def identify_trossen_robot_visually(robot: Robot) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position"""
    if not isinstance(robot, TrossenSingleArmRobot):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    driver = trossen_arm.TrossenArmDriver()

    logger.info("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        robot.payload.connection_string,
        True,
        timeout=5,
    )

    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.0, 0.5, True)


async def _identify_arm(connection_string: str, label: str) -> None:
    """Identify a single arm by its IP address."""
    driver = trossen_arm.TrossenArmDriver()
    logger.info(f"Configuring {label} arm driver at {connection_string}...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        connection_string,
        True,
        timeout=5,
    )
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.0, 0.5, True)


async def identify_trossen_bimanual_robot_visually(robot: Robot) -> None:
    """Identify both arms of a bimanual robot sequentially."""
    if not isinstance(robot, TrossenBimanualRobot):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    await _identify_arm(robot.payload.connection_string_left, "left")
    await _identify_arm(robot.payload.connection_string_right, "right")

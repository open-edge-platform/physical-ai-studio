import trossen_arm

from schemas import Robot


async def identify_trossen_robot_visually(robot: Robot) -> None:
    """Identify the robot by moving the joint from current to min to max to initial position"""
    if not robot.type.lower().startswith("trossen"):
        raise ValueError(f"Trying to identify unsupported robot: {robot.type}")

    driver = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        robot.connection_string,
        True,
        timeout=5
    )

    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.02, 0.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(0.0, 0.5, True)

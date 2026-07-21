from physicalai.robot.trossen import BimanualWidowXAI, WidowXAI

from schemas.robot import RobotType, TrossenBimanualPayload, TrossenSingleArmPayload

from .types import CatalogRobot, CatalogRobotFactory, RobotAdapterOptions, RobotCatalogDefinition

_TROSSEN_TO_URDF = {
    "shoulder_pan.pos": ["joint_0"],
    "shoulder_lift.pos": ["joint_1"],
    "elbow_flex.pos": ["joint_2"],
    "wrist_flex.pos": ["joint_3"],
    "wrist_yaw.pos": ["joint_4"],
    "wrist_roll.pos": ["joint_5"],
    "gripper.pos": ["left_carriage_joint", "right_carriage_joint"],
}

_BIMANUAL_TROSSEN_TO_URDF = {
    "left_shoulder_pan.pos": ["follower_left_joint_0"],
    "left_shoulder_lift.pos": ["follower_left_joint_1"],
    "left_elbow_flex.pos": ["follower_left_joint_2"],
    "left_wrist_flex.pos": ["follower_left_joint_3"],
    "left_wrist_yaw.pos": ["follower_left_joint_4"],
    "left_wrist_roll.pos": ["follower_left_joint_5"],
    "left_gripper.pos": ["follower_left_left_carriage_joint", "follower_left_right_carriage_joint"],
    "right_shoulder_pan.pos": ["follower_right_joint_0"],
    "right_shoulder_lift.pos": ["follower_right_joint_1"],
    "right_elbow_flex.pos": ["follower_right_joint_2"],
    "right_wrist_flex.pos": ["follower_right_joint_3"],
    "right_wrist_yaw.pos": ["follower_right_joint_4"],
    "right_wrist_roll.pos": ["follower_right_joint_5"],
    "right_gripper.pos": ["follower_right_left_carriage_joint", "follower_right_right_carriage_joint"],
}


async def _build_trossen_single_arm_driver(
    robot: CatalogRobot[TrossenSingleArmPayload], _factory: CatalogRobotFactory
) -> WidowXAI:
    role = "follower" if robot.type == RobotType.TROSSEN_WIDOWXAI_FOLLOWER else "leader"
    return WidowXAI(ip=robot.payload.connection_string, role=role)


async def _build_trossen_bimanual_driver(
    robot: CatalogRobot[TrossenBimanualPayload], _factory: CatalogRobotFactory
) -> BimanualWidowXAI:
    mode = "follower" if robot.type == RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER else "leader"
    left_driver = WidowXAI(ip=robot.payload.connection_string_left, role=mode)
    right_driver = WidowXAI(ip=robot.payload.connection_string_right, role=mode)
    return BimanualWidowXAI(left=left_driver, right=right_driver)


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in WidowX AI robot catalog definitions."""
    return [
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_WIDOWXAI_FOLLOWER,
            display_name="Trossen WidowX AI Follower",
            role="follower",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_FOLLOWER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_FOLLOWER}",
            },
            joint_map=_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/wxai/wxai_follower.urdf",
            robot_builder=_build_trossen_single_arm_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_WIDOWXAI_LEADER,
            display_name="Trossen WidowX AI Leader",
            role="leader",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_LEADER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_WIDOWXAI_LEADER}",
            },
            joint_map=_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/wxai/wxai_follower.urdf",
            robot_builder=_build_trossen_single_arm_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER,
            display_name="Trossen Bimanual WidowX AI Follower",
            role="follower",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_FOLLOWER}",
            },
            joint_map=_BIMANUAL_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/stationary_ai.urdf",
            robot_builder=_build_trossen_bimanual_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
        ),
        RobotCatalogDefinition(
            type=RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER,
            display_name="Trossen Bimanual WidowX AI Leader",
            role="leader",
            urdf_path=f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER}/urdf",
            package_map={
                "trossen_arm_description": f"/api/robots/catalog/{RobotType.TROSSEN_BIMANUAL_WIDOWXAI_LEADER}",
            },
            joint_map=_BIMANUAL_TROSSEN_TO_URDF,
            urdf_relative_path="widowx/urdf/generated/stationary_ai.urdf",
            robot_builder=_build_trossen_bimanual_driver,
            adapter_options=RobotAdapterOptions(include_velocities=True, goal_time_scale=1.0, external_effort_gain=0.1),
        ),
    ]

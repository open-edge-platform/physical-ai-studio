from schemas.robot import RobotType

from .types import RobotCatalogDefinition

_SO101_TO_URDF = {
    "shoulder_pan.pos": ["shoulder_pan"],
    "shoulder_lift.pos": ["shoulder_lift"],
    "elbow_flex.pos": ["elbow_flex"],
    "wrist_flex.pos": ["wrist_flex"],
    "wrist_roll.pos": ["wrist_roll"],
    "gripper.pos": ["gripper"],
}


def get_definitions() -> list[RobotCatalogDefinition]:
    """Return built-in SO101 robot catalog definitions."""
    urdf_relative_path = "SO101/so101_new_calib.urdf"

    return [
        RobotCatalogDefinition(
            type=RobotType.SO101_FOLLOWER,
            display_name="SO101 Follower",
            role="follower",
            urdf_path=f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}/urdf",
            package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_FOLLOWER}"},
            joint_map=_SO101_TO_URDF,
            urdf_relative_path=urdf_relative_path,
        ),
        RobotCatalogDefinition(
            type=RobotType.SO101_LEADER,
            display_name="SO101 Leader",
            role="leader",
            urdf_path=f"/api/robots/catalog/{RobotType.SO101_LEADER}/urdf",
            package_map={"SO101": f"/api/robots/catalog/{RobotType.SO101_LEADER}"},
            joint_map=_SO101_TO_URDF,
            urdf_relative_path=urdf_relative_path,
        ),
    ]

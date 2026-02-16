from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors

from robots.utils import get_robot_client
from schemas.environment import EnvironmentWithRelations
from schemas.project_camera import Camera
from schemas.robot import RobotType
from services.robot_calibration_service import RobotCalibrationService
from utils.serial_robot_tools import RobotConnectionManager


async def build_observation_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
) -> dict:
    """Return dict of action features of environment."""
    if len(environment.robots) > 1:
        # TODO: Implement, should probably prefix feature the robots only when len(robots) > 1
        # One issue is that you need to know which is which, so probably need a name identifier for robots
        raise ValueError("Environments with multiple robots not implemented yet")
    output_features = await build_action_features(environment, robot_manager, calibration_service)
    for camera in environment.cameras:
        output_features[str(camera.id)] = await get_camera_features(camera)

    return output_features


def robot_for_action_features(action_features: list[str]) -> RobotType:
    """Todo: Do this proper. This is a bad idea"""
    if len(action_features) >= 7:
        return RobotType.TROSSEN_WIDOWXAI_FOLLOWER
    return RobotType.SO101_FOLLOWER


async def build_action_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
) -> dict:
    """Return dict of action features of environment."""
    output_features = {}
    for robot in environment.robots:
        client = await get_robot_client(robot.robot, robot_manager, calibration_service)
        for feature in client.features():
            output_features[feature] = float
    return output_features


async def build_lerobot_dataset_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
    use_videos: bool = True,
) -> dict:
    """Build lerobot dataset features."""
    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    observation_features = await build_observation_features(environment, robot_manager, calibration_service)
    action_features = await build_action_features(environment, robot_manager, calibration_service)

    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=use_videos,
        ),
    )


async def get_camera_features(camera: Camera) -> tuple[int, int, int]:
    """Get features of a camera.

    Note: This works for 'now', but ip cameras etc should probably just get a frame before returning this.
    """
    if camera.payload is None or camera.payload.height is None or camera.payload.width is None:
        raise ValueError("Cannot get features of camera without payload.")
    return (camera.payload.height, camera.payload.width, 3)

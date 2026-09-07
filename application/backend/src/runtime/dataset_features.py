from __future__ import annotations

from typing import Any

from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.utils.feature_utils import combine_feature_dicts

from runtime.features import feature_names


def build_lerobot_dataset_features(
    *,
    joint_names: list[str],
    camera_specs: dict[str, tuple[int, int, int]],
    use_videos: bool = True,
) -> dict[str, Any]:
    """Build LeRobot feature metadata from joint names and live frame shapes.

    Camera keys must already be the sanitized names ``runtime.config_builder``
    uses for the runtime camera mapping — those are the keys a trained model
    will look up. Dimensions come from ``Frame.data.shape``, not the database
    row, so a camera whose row still says 640x480 but whose publisher is
    1280x720 is recorded at the size the session actually saw.
    """
    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()

    action_features: dict[str, Any] = {}
    observation_features: dict[str, Any] = {}
    for feature in feature_names(joint_names, include_velocities=False):
        action_features[feature] = float
        observation_features[feature] = float

    for camera_key, spec in camera_specs.items():
        observation_features[camera_key] = spec

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

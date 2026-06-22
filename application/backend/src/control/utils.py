import base64
from typing import Any

import cv2
import numpy as np

from control.data_registry import EnvironmentDataRegistry
from workers.camera_worker import CameraWorker


def build_lerobot_dataset_features(manifest: EnvironmentDataRegistry, use_videos: bool = True) -> dict:
    """Build lerobot dataset features."""
    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors

    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    action_features: dict[str, Any] = {}
    observation_features: dict[str, Any] = {}
    for feature in manifest.robot.features:
        action_features[feature] = float
        observation_features[feature] = float

    for camera in manifest.cameras:
        observation_features[camera.name.lower()] = (camera.height, camera.width, 3)

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


def get_observation_from_manifest(manifest: EnvironmentDataRegistry, timestamp: float = 0) -> dict:
    """Lightweight read-only get data from environments SharedMemory."""
    actions = list(manifest.robot.actions.get_obj())
    state = list(manifest.robot.state.get_obj())

    camera_images = {}
    for camera in manifest.cameras:
        frame = CameraWorker.frame_from_buffer(camera.frame_data.get_obj(), camera.width, camera.height)
        camera_images[camera.id] = frame

    return {
        "state": state,
        "action": actions,
        "images": camera_images,
        "timestamp": timestamp,
    }


def format_observation_for_dataset(observation: dict, manifest: EnvironmentDataRegistry) -> tuple[dict, dict]:
    """Format observation for input of a dataset."""
    result = {i: observation["state"][k] for k, i in enumerate(manifest.robot.features)}
    actions = {i: observation["action"][k] for k, i in enumerate(manifest.robot.features)}
    for camera in manifest.cameras:
        camera_name = camera.name.lower()
        result[camera_name] = np.ascontiguousarray(observation["images"][camera.id])

    return result, actions


def format_observation_for_model(observation: dict, manifest: EnvironmentDataRegistry, task: str) -> Any:
    """Format observation dict into a model-ready Observation object."""
    from physicalai.data import Observation

    images: dict = {}
    for camera in manifest.cameras:
        camera_name = camera.name.lower()
        # HWC → CHW, float 0..1 range.
        images[camera_name] = np.ascontiguousarray(
            observation["images"][camera.id].transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255
        )

    return Observation(
        state=np.array([observation["state"]], dtype=np.float32),
        images=images,
        task=task,  # type: ignore[bad-argument-type]  # TODO: Implement tasks.
    )


def format_observation_for_reporting(observation: dict, manifest: EnvironmentDataRegistry) -> dict:
    """Format observation for UI."""
    actions = {i: observation["action"][k] for k, i in enumerate(manifest.robot.features)}
    state = {i: observation["state"][k] for k, i in enumerate(manifest.robot.features)}
    camera_images = {}
    for camera in manifest.cameras:
        frame = np.ascontiguousarray(observation["images"][camera.id][..., ::-1])  # RGB→BGR for cv2
        success, imagebytes = cv2.imencode(".jpg", frame)
        if success:
            camera_images[camera.id] = base64.b64encode(imagebytes).decode()

    return {
        "state": state,
        "actions": actions,
        "cameras": camera_images,
        "timestamp": observation["timestamp"],
    }

import asyncio
import base64
from typing import Any

import cv2
import numpy as np
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from loguru import logger
from physicalai.data import Observation

from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations, TeleoperatorRobotWithRobot
from schemas.project_camera import Camera
from utils.async_camera_capture import AsyncCameraCapture
from workers.camera_worker import create_frames_source_from_camera


class EnvironmentIntegration:
    """Integration class for the inference version of an environment."""

    environment: EnvironmentWithRelations
    robot_client_factory: RobotClientFactory
    action_keys: list[str] = []
    follower: RobotClient | None = None
    leader: RobotClient | None = None
    frame_captures: dict[str, AsyncCameraCapture]

    def __init__(self, environment: EnvironmentWithRelations, robot_client_factory: RobotClientFactory):
        self.environment = environment
        self.robot_client_factory = robot_client_factory
        self.frame_captures = {}

    async def setup(self) -> None:
        robot = self.environment.robots[0]  # TODO: Handle multiple robots?

        self.follower = await self.robot_client_factory.build(robot.robot)
        if isinstance(robot.tele_operator, TeleoperatorRobotWithRobot) and robot.tele_operator.robot is not None:
            self.leader = await self.robot_client_factory.build(robot.tele_operator.robot)
        self.action_keys = self.follower.features()

        self.frame_captures = {}
        for cam_cfg in self.environment.cameras:
            cam_id = str(cam_cfg.id)
            cam = create_frames_source_from_camera(cam_cfg)  # gives you the object with connect/read

            cap = AsyncCameraCapture(
                camera=cam,
                fps=cam_cfg.payload.fps,
                use_cached_on_failure=True,
            )
            await cap.start()
            self.frame_captures[cam_id] = cap

        await asyncio.sleep(1)  # sleep for camera warmup
        await self.follower.connect()
        if self.leader:
            await self.leader.connect()

    def build_lerobot_dataset_features(self, use_videos: bool = True) -> dict:
        """Build lerobot dataset features."""
        teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()

        if self.follower is None:
            raise ValueError("Can only build features with follower")
        action_features: dict[str, Any] = {}
        observation_features: dict[str, Any] = {}
        for feature in self.follower.features():
            action_features[feature] = float
            observation_features[feature] = float

        for camera in self.environment.cameras:
            observation_features[camera.name.lower()] = self._get_camera_features(camera)

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

    async def set_joints_state(self, actions: dict, goal_time: float) -> None:
        """Set joints on robot"""
        if self.follower:
            await self.follower.set_joints_state(actions, goal_time)

    async def get_observation(self) -> dict | None:
        if self.follower and self.frame_captures:
            observation = (await self.follower.read_state())["state"]

            for cam_id, cap in self.frame_captures.items():
                frame, t_perf, ok, err, seq = await cap.get_latest()
                if ok and frame is not None:
                    observation[cam_id] = frame

            return observation

        return None

    async def set_follower_position_from_leader(self, goal_time: float) -> dict | None:
        """Directly set the follower position based on leader."""
        if self.leader is not None:
            actions = (await self.leader.read_state())["state"]
            await self.set_joints_state(actions, goal_time)

            if self.follower and self.leader:
                forces = await self.follower.read_forces()
                if forces and forces["state"] is not None:
                    await self.leader.set_forces(forces["state"])

            return actions
        return None

    def format_observation_for_reporting(self, observation: dict, timestamp: float) -> dict:
        camera_keys = [str(camera.id) for camera in self.environment.cameras]
        camera_images = {camera: self._base_64_encode_observation(observation[camera]) for camera in camera_keys}

        return {
            "state": {key: observation[key] for key in self.action_keys},
            "actions": None,
            "cameras": camera_images,
            "timestamp": timestamp,
        }

    def format_model_input_observation(self, raw_observation: dict, task: str | None = None) -> Observation:  # noqa: ARG002
        observation = self._remap_camera_observations(raw_observation)
        state = np.array([[value for key, value in observation.items() if key in self.action_keys]], dtype=np.float32)
        images: dict = {}
        for camera in self.environment.cameras:
            camera_name = camera.name.lower()
            # SWAP HWC, RGB2BGR and in float 0..1 range.
            images[camera_name] = np.ascontiguousarray(
                observation[camera_name][..., ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255
            )

        return Observation(
            state=state,
            images=images,
            # task=task, # TODO: Implement tasks.
        )

    def format_observation_for_dataset(self, raw_observation: dict) -> dict:
        """Format observation for dataset frame input."""
        observation = self._remap_camera_observations(raw_observation)
        for camera in self.environment.cameras:
            camera_name = camera.name.lower()
            # RGB2BGR
            observation[camera_name] = np.ascontiguousarray(observation[camera_name][..., ::-1])

        return observation

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()

    def _remap_camera_observations(self, observations: dict) -> dict:
        """Remap camera observations from camera ID keys to lowercase camera name keys."""
        lerobot_observations = dict(observations)
        for camera in self.environment.cameras:
            lerobot_observations[camera.name.lower()] = lerobot_observations.pop(str(camera.id))
        return lerobot_observations

    @staticmethod
    def _get_camera_features(camera: Camera) -> tuple[int, int, int]:
        """Get features of a camera.

        Note: This works for 'now', but ip cameras etc should probably just get a frame before returning this.
        """
        if camera.payload is None or camera.payload.height is None or camera.payload.width is None:
            raise ValueError("Cannot get features of camera without payload.")
        return (camera.payload.height, camera.payload.width, 3)

    async def teardown(self) -> None:
        if self.follower:
            await self.follower.disconnect()

        if self.leader:
            await self.leader.disconnect()

        for cap in self.frame_captures.values():
            try:
                await cap.stop()
            except Exception:
                logger.info("Failed stopping a camera thread. Ignoring")

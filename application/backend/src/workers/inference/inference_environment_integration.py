import asyncio
import base64

import cv2
import numpy as np
from frame_source.video_capture_base import VideoCaptureBase
from physicalai.data import Observation

from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations
from workers.camera_worker import create_frames_source_from_camera


class InferenceEnvironmentIntegration:
    """Integration class for the inference version of an environment."""

    environment: EnvironmentWithRelations
    robot_client_factory: RobotClientFactory
    action_keys: list[str] = []
    follower: RobotClient | None = None
    cameras: dict[str, VideoCaptureBase] | None = None

    def __init__(self, environment: EnvironmentWithRelations, robot_client_factory: RobotClientFactory):
        self.environment = environment
        self.robot_client_factory = robot_client_factory

    async def setup(self) -> None:
        robot = self.environment.robots[0]  # TODO: Handle multiple robots?

        self.follower = await self.robot_client_factory.build(robot.robot)
        self.action_keys = self.follower.features()

        self.cameras = {str(camera.id): create_frames_source_from_camera(camera) for camera in self.environment.cameras}

        for camera in self.cameras.values():
            camera.connect()
            camera.start_async()

        await asyncio.sleep(1)  # sleep for camera warmup. TODO: Refactor start_async to proper camera wrapper
        await self.follower.connect()

    async def set_joints_state(self, actions: dict, goal_time: float) -> None:
        """Set joints on robot"""
        if self.follower:
            await self.follower.set_joints_state(actions, goal_time)

    async def get_observation(self) -> dict | None:
        if self.follower and self.cameras:
            observation = (await self.follower.read_state())["state"]
            for camera_id, camera in self.cameras.items():
                _success, camera_frame = camera.read()  # HWC
                if camera_frame is None:
                    raise Exception("Camera frame is None")
                observation[camera_id] = camera_frame

            return observation

        return None

    def format_observation_for_reporting(self, observation: dict, timestamp: float) -> dict:
        if self.cameras:
            camera_images = {camera: self._base_64_encode_observation(observation[camera]) for camera in self.cameras}

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

    async def teardown(self) -> None:
        if self.follower:
            await self.follower.disconnect()
        if self.cameras:
            for camera in self.cameras.values():
                camera.stop()

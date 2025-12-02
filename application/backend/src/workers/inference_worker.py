import base64
import time
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

import cv2
import numpy as np
import torch
from getiaction.data import Observation
from getiaction.policies import ACT, ACTModel
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import busy_wait
from loguru import logger

from schemas import InferenceConfig
from utils.camera import build_camera_config
from utils.framesource_bridge import FrameSourceCameraBridge
from utils.robot import make_lerobot_robot_config_from_robot

from .base import BaseThreadWorker

SO_101_REST_POSITION = {
    "shoulder_pan.pos": -2,
    "shoulder_lift.pos": -90,
    "elbow_flex.pos": 100,
    "wrist_flex.pos": 60,
    "wrist_roll.pos": 0,
    "gripper.pos": 25,
}


class InferenceWorker(BaseThreadWorker):
    ROLE: str = "InferenceWorker"

    events: dict[str, EventClass]
    queue: Queue
    is_running = False

    action_keys: list[str] = []
    camera_keys: list[str] = []

    def __init__(self, stop_event: EventClass, queue: Queue, config: InferenceConfig):
        super().__init__(stop_event=stop_event)
        self.config = config
        self.queue = queue
        self.events = {
            "stop": Event(),
            "calculate_trajectory": Event(),
            "start": Event(),
            "disconnect": Event(),
        }

    def calculate_trajectory(self) -> None:
        """Calculate trajectory."""
        self.events["calculate_trajectory"].set()

    def start_task(self, task_index: int) -> None:
        """Start specific task index"""
        self.config.task_index = task_index
        self.events["start"].set()

    def stop(self) -> None:
        """Stop inference."""
        self.events["stop"].set()

    def disconnect(self) -> None:
        """Stop inference and teardown."""
        self.events["disconnect"].set()

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        logger.info("connect to robot, cameras and setup dataset")
        cameras = {camera.name: build_camera_config(camera) for camera in self.config.cameras}
        follower_config = make_lerobot_robot_config_from_robot(self.config.robot, cameras)
        # follower_config.max_relative_target = 1

        model_path = self.config.model.path
        self.robot = make_robot_from_config(follower_config)
        logger.info(f"loading model: {model_path}")
        self.model = ACTModel.load_from_checkpoint(Path(model_path))
        self.model.eval()

        self.policy = ACT(self.model)
        self.policy.eval()

        # TODO: Define this somehow
        # LeRobot tends to return the robot arm to root position on reset.
        # This seems to work better for act when I change the environment mid inference

        # After setting up the robot, instantiate the FrameSource using a bridge
        # This can be done directly once switched over to LeRobotDataset V3.
        # We do need to first instantiate using the lerobot dict because a follower requires cameras.
        self.robot.cameras = {camera.name: FrameSourceCameraBridge(camera) for camera in self.config.cameras}
        self.robot.connect()

        self.robot.send_action(SO_101_REST_POSITION)

        self.action_keys = [f"{key}.pos" for key in self.robot.bus.sync_read("Present_Position")]
        self.camera_keys = [camera.name for camera in self.config.cameras]

        logger.info("inference all setup, reporting state")
        self._report_state()

    async def run_loop(self) -> None:
        """inference loop."""
        logger.info("run loop")
        self.events["start"].clear()
        self.events["stop"].clear()
        self.events["disconnect"].clear()

        self.is_running = False

        start_episode_t = time.perf_counter()
        action_queue: list[list[float]] = []
        while not self.should_stop() and not self.events["disconnect"].is_set():
            start_loop_t = time.perf_counter()
            if self.events["start"].is_set():
                logger.info("start")
                self.events["start"].clear()
                self.robot.send_action(SO_101_REST_POSITION)
                busy_wait(0.3)  # TODO check if neccesary
                self.is_running = True
                start_episode_t = time.perf_counter()
                self._report_state()

            if self.events["stop"].is_set():
                logger.info("stop")
                self.events["stop"].clear()
                action_queue.clear()
                busy_wait(0.3)  # TODO check if neccesary
                self.is_running = False
                self._report_state()

            if self.events["calculate_trajectory"].is_set():
                logger.info("calculate_trajectory")
                self.events["calculate_trajectory"].clear()
                lerobot_obs = self.robot.get_observation()
                observation = self._build_geti_action_observation(lerobot_obs)
                logger.info(observation.keys())
                self._report_trajectory(self.model(observation.to_dict())[0].tolist())

            lerobot_obs = self.robot.get_observation()
            timestamp = time.perf_counter() - start_episode_t
            if self.is_running:
                observation = self._build_geti_action_observation(lerobot_obs)
                if not action_queue:
                    action_queue = self.model(observation.to_dict())[0].tolist()
                action = action_queue.pop(0)

                # print(observation)
                # actions = self.policy.select_action(observation)
                formatted_actions = dict(zip(self.action_keys, action))
                self.robot.send_action(formatted_actions)
                self._report_action(formatted_actions, lerobot_obs, timestamp)
            else:
                self._report_action({}, lerobot_obs, timestamp)

            dt_s = time.perf_counter() - start_loop_t
            wait_time = 1 / self.config.fps - dt_s

            busy_wait(wait_time)

    def teardown(self) -> None:
        """Disconnect robots and close queue."""
        self.robot.disconnect()
        self.queue.close()
        self.queue.cancel_join_thread()

    def _report_state(self):
        state = {
            "event": "state",
            "data": {"initialized": True, "is_running": self.is_running, "task_index": self.config.task_index},
        }
        logger.info(f"inference state: {state}")
        self.queue.put(state)

    def _report_trajectory(self, trajectory: list[dict]):
        self.queue.put({"event": "trajectory", "data": {"trajectory": trajectory}})

    def _report_action(self, actions: dict, observation: dict, timestamp: float):
        """Report observation to queue."""
        self.queue.put(
            {
                "event": "observations",
                "data": {
                    "actions": actions,
                    "state": {key: observation.get(key, 0) for key in self.action_keys},
                    "cameras": {
                        key: self._base_64_encode_observation(cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
                        for key in self.camera_keys
                        if key in observation
                    },
                    "timestamp": timestamp,
                },
            }
        )

    def _build_geti_action_observation(self, robot_observation: dict):
        state = torch.tensor([value for key, value in robot_observation.items() if key in self.action_keys]).unsqueeze(
            0
        )
        images: dict = {}
        for name in self.camera_keys:
            frame = robot_observation[name]

            # change image to 0..1 and swap R & B channels.
            images[name] = torch.from_numpy(frame)
            images[name] = images[name].float() / 255
            images[name] = images[name].permute(2, 0, 1).contiguous()
            images[name] = images[name].unsqueeze(0)

        return Observation(
            state=state,
            images=images,
        )

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()

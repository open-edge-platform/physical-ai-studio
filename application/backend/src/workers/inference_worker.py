import base64
from pathlib import Path
import shutil
import time
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

import cv2
from getiaction.data import Observation
from getiaction.policies import ACT, ACTModel
import numpy as np
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import busy_wait
from loguru import logger
import torch

from schemas import InferenceConfig
from utils.camera import build_camera_config
from utils.robot import make_lerobot_robot_config_from_robot

from .base import BaseProcessWorker


class InferenceWorker(BaseProcessWorker):
    ROLE: str = "InferenceWorker"

    events: dict[str, EventClass]
    queue: Queue
    is_running = False

    action_keys: list[str] = []
    camera_keys: list[str] = []

    def __init__(self, stop_event: EventClass, queue: Queue, config: InferenceConfig):
        super().__init__(stop_event=stop_event, queues_to_cancel=[queue])
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
        #follower_config.max_relative_target = 1

        model_path = "/home/ronald/intel/geti-action/application/backend/act_policy_real_data.pt"
        self.robot = make_robot_from_config(follower_config)
        logger.info(f"loading model: {model_path}")
        self.model = ACTModel.load_from_checkpoint(Path(model_path))
        self.model.eval()

        self.policy = ACT(self.model)
        self.policy.eval()

        #TODO: Define this somehow
        # LeRobot tends to return the robot arm to root position on reset.
        # This seems to work better for act when I change the environment mid inference
        self.root_position_action = {'shoulder_pan.pos': -2.271006813020435, 'shoulder_lift.pos': -98.08027923211169, 'elbow_flex.pos': 99.37527889335118, 'wrist_flex.pos': 67.34527687296418, 'wrist_roll.pos': -13.406593406593402, 'gripper.pos': 27.128953771289538}


        # After setting up the robot, instantiate the FrameSource using a bridge
        # This can be done directly once switched over to LeRobotDataset V3.
        # We do need to first instantiate using the lerobot dict because a follower requires cameras.
        # self.robot.cameras = {camera.name: FrameSourceCameraBridge(camera) for camera in self.config.cameras}

        self.robot.connect()

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
        frame_index = 0
        while not self.should_stop() and not self.events["disconnect"].is_set():
            start_loop_t = time.perf_counter()
            if self.events["start"].is_set():
                logger.info("start")
                self.events["start"].clear()
                self.robot.send_action(self.root_position_action)
                busy_wait(0.3)  # TODO check if neccesary
                self.is_running = True
                start_episode_t = time.perf_counter()
                frame_index = 0
                self._report_state()

            if self.events["stop"].is_set():
                logger.info("stop")
                self.events["stop"].clear()
                busy_wait(0.3)  # TODO check if neccesary
                self.is_running = False
                self._report_state()

            if self.events["calculate_trajectory"].is_set():
                logger.info("calculate_trajectory")
                self.events["calculate_trajectory"].clear()
                lerobot_obs = self.robot.get_observation()
                observation = self._build_geti_action_observation(lerobot_obs, 0, 0)
                logger.info(observation.keys())
                self._report_trajectory(self.model(observation.to_dict())[0].tolist())

            if self.is_running:

                timestamp = time.perf_counter() - start_episode_t
                lerobot_obs = self.robot.get_observation()
                observation = self._build_geti_action_observation(lerobot_obs, timestamp, frame_index)
                #print(observation)
                actions = self.policy.select_action(observation)
                formatted_actions = dict(zip(self.action_keys, actions[0].tolist()))
                self._report_action(formatted_actions, lerobot_obs, timestamp)
                self.robot.send_action(formatted_actions)
                frame_index += 1

            dt_s = time.perf_counter() - start_loop_t
            wait_time = 1 / self.config.fps - dt_s

            busy_wait(wait_time)

    def teardown(self) -> None:
        """Disconnect robots and close queue."""
        self.robot.disconnect()
        self.queue.close()

    def _report_state(self):
        state = {"event": "state", "data": {"initialized": True, "is_running": self.is_running, "task_index": self.config.task_index}}
        logger.info(f"inference state: {state}")
        self.queue.put(state)

    def _report_trajectory(self, trajectory: list[dict]):
        self.queue.put(
            {
                "event": "trajectory",
                "data": {
                    "trajectory": trajectory
                }
            }
        )


    def _report_action(self, actions: dict, observation: dict, timestamp: float):
        """Report observation to queue."""
        self.queue.put(
            {
                "event": "observations",
                "data": {
                    "actions": actions,
                    "cameras": {
                        key: self._base_64_encode_observation(observation.get(key))
                        for key in self.camera_keys
                        if key in observation
                    },
                    "timestamp": timestamp,
                },
            }
        )

    def _build_geti_action_observation(self, robot_observation: dict, timestamp: float, frame_index: int):
        state = torch.tensor([value for key, value in robot_observation.items() if key in self.action_keys ]).unsqueeze(0)
        images: dict = {}
        for name in self.camera_keys:
            #key = f"observation.images.{name}"
            frame = robot_observation[name]
            logger.info(frame.shape)
            logger.info(frame.dtype)

            images[name] = torch.from_numpy(frame)
            images[name] = images[name].float() / 255
            images[name] = images[name].permute(2, 0, 1).contiguous()
            images[name] = images[name].unsqueeze(0)

        return Observation(
            state=state,
            images=images,
            task_index=torch.tensor([0]),
            task=["Place block on paper"],
            timestamp=torch.tensor([timestamp]),
            frame_index=torch.tensor([frame_index]),
        )

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()

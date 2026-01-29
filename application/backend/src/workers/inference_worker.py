import asyncio
import base64
import time
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

import cv2
import numpy as np
import torch
from getiaction.data import Observation
from getiaction.inference import InferenceModel
from getiaction.policies import ACT
from lerobot.utils.robot_utils import precise_sleep
from loguru import logger
from pydantic import BaseModel

from robots.robot_client import RobotClient
from robots.utils import get_robot_client
from schemas import InferenceConfig
from services.robot_calibration_service import RobotCalibrationService
from utils.robot import RobotConnectionManager
from workers.camera_worker import create_frames_source_from_camera

from .base import BaseThreadWorker

SO_101_REST_POSITION = {
    "shoulder_pan.pos": -2,
    "shoulder_lift.pos": -90,
    "elbow_flex.pos": 100,
    "wrist_flex.pos": 60,
    "wrist_roll.pos": 0,
    "gripper.pos": 25,
}


class InferenceState(BaseModel):
    is_running: bool = False
    task_index: int = 0
    initialized: bool = False
    error: bool = False


class InferenceWorker(BaseThreadWorker):
    ROLE: str = "InferenceWorker"

    robot_manager: RobotConnectionManager
    calibration_service: RobotCalibrationService

    events: dict[str, EventClass]
    queue: Queue
    state: InferenceState

    follower: RobotClient
    action_keys: list[str] = []
    camera_keys: list[str] = []

    def __init__(
        self,
        stop_event: EventClass,
        queue: Queue,
        config: InferenceConfig,
        calibration_service: RobotCalibrationService,
        robot_manager: RobotConnectionManager,
    ):
        super().__init__(stop_event=stop_event)
        self.config = config
        self.queue = queue
        self.state = InferenceState()
        self.calibration_service = calibration_service
        self.robot_manager = robot_manager
        self.events = {
            "stop": Event(),
            "start": Event(),
            "disconnect": Event(),
        }

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

    async def setup_environment(self) -> None:
        """Setup environment."""

        robot = self.config.environment.robots[0]  # Assume 1 arm for now.
        self.follower = await get_robot_client(robot.robot, self.robot_manager, self.calibration_service)
        self.cameras = {
            camera.name: create_frames_source_from_camera(camera) for camera in self.config.environment.cameras
        }
        for camera in self.cameras.values():
            # camera.attach_processor(CameraFrameProcessor()) # TODO Not working. Fix in framesource
            camera.connect()

        await self.follower.connect()

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        logger.info("connect to robot, cameras and setup dataset")
        try:
            asyncio.run(self.setup_environment())

            model_path = Path(self.config.model.path)
            logger.info(f"loading model: {model_path}")
            policy = ACT.load_from_checkpoint(str(model_path / "model.ckpt"))
            export_dir = model_path / "exports" / self.config.backend
            if not export_dir.is_dir():
                policy.export(export_dir, backend=self.config.backend)

            self.model = InferenceModel(
                export_dir=export_dir, policy_name=self.config.model.policy, backend=self.config.backend
            )

            self.follower.set_joints_state(SO_101_REST_POSITION)

            self.action_keys = self.follower.features()
            self.camera_keys = list(self.cameras)

            self.state.initialized = True
            logger.info("inference all setup, reporting state")
        except Exception as e:
            self.state.error = True
            self._report_error(e)
        self._report_state()

    async def run_loop(self) -> None:
        """inference loop."""
        try:
            logger.info("run loop")
            self.events["start"].clear()
            self.events["stop"].clear()
            self.events["disconnect"].clear()

            self.state.is_running = False

            start_episode_t = time.perf_counter()
            action_queue: list[list[float]] = []
            while not self.should_stop() and not self.events["disconnect"].is_set():
                if not self.state.initialized or self.state.error:
                    return

                start_loop_t = time.perf_counter()
                if self.events["start"].is_set():
                    logger.info("start")
                    self.events["start"].clear()
                    self.follower.set_joints_state(SO_101_REST_POSITION)
                    precise_sleep(0.3)  # TODO check if neccesary
                    self.state.is_running = True
                    start_episode_t = time.perf_counter()
                    self._report_state()

                if self.events["stop"].is_set():
                    logger.info("stop")
                    self.events["stop"].clear()
                    action_queue.clear()
                    precise_sleep(0.3)  # TODO check if neccesary
                    self.state.is_running = False
                    action_queue.clear()
                    self._report_state()

                state = (await self.follower.read_state())["state"]
                timestamp = time.perf_counter() - start_episode_t
                logger.info(f"{timestamp}, {state}")
                if self.state.is_running:
                    # TODO: Implement for new environment
                    pass
                    # observation = self._build_geti_action_observation(lerobot_obs)
                    # if not action_queue:
                    #    action_queue = self.model.select_action(observation)[0].tolist()
                    # action = action_queue.pop(0)

                    # print(observation)
                    # formatted_actions = dict(zip(self.action_keys, action))
                    # self.robot.send_action(formatted_actions)
                    # self._report_action(formatted_actions, lerobot_obs, timestamp)
                else:
                    pass
                    # self._report_action({}, lerobot_obs, timestamp)
                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / 30 - dt_s

                precise_sleep(wait_time)
        except Exception as e:
            self._report_error(e)

    def teardown(self) -> None:
        """Disconnect robots and close queue."""
        if self.follower.is_connected:
            asyncio.run(self.follower.disconnect())

        # Wait for .5 seconds before closing queue to allow messages thru
        asyncio.run(asyncio.sleep(0.5))
        self.queue.close()
        self.queue.cancel_join_thread()

    def _report_state(self):
        state = {"event": "state", "data": self.state.model_dump()}
        logger.info(f"inference state: {state}")
        self.queue.put(state)

    def _report_error(self, error: BaseException):
        data = {
            "event": "error",
            "data": str(error),
        }
        logger.error(f"error: {data}")
        self.queue.put(data)

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

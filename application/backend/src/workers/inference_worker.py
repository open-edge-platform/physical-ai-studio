# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import time
import traceback
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

import cv2
import numpy as np
import torch
from getiaction.data import Observation
from lerobot.utils.robot_utils import precise_sleep
from loguru import logger
from pydantic import BaseModel

from models.utils import load_inference_model
from robots.robot_client import RobotClient
from robots.utils import get_robot_client
from schemas import InferenceConfig
from services.robot_calibration_service import RobotCalibrationService
from utils.serial_robot_tools import RobotConnectionManager
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


class CameraFrameProcessor:
    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        """Post process camera frame."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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
        self.action_queue: list[list[float]] = []

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
            camera.start_async()

        await asyncio.sleep(1)  # sleep for camera warmup. TODO: Refactor start_async to proper camera wrapper
        await self.follower.connect()

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        logger.info("connect to robot, cameras and setup dataset")
        try:
            if self.loop is None:
                raise RuntimeError("The event loop must be set.")
            self.loop.run_until_complete(self.setup_environment())

            self.model = load_inference_model(self.config.model, backend=self.config.backend)

            self.action_keys = self.follower.features()
            self.camera_keys = list(self.cameras)

            for camera in self.cameras.values():
                camera.start_async()

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

            while not self.should_stop() and not self.events["disconnect"].is_set():
                if not self.state.initialized or self.state.error:
                    return

                start_loop_t = time.perf_counter()
                if self.events["start"].is_set():
                    start_episode_t = time.perf_counter()
                    await self._on_start()

                if self.events["stop"].is_set():
                    await self._on_stop()

                state = (await self.follower.read_state())["state"]
                for camera_name, camera in self.cameras.items():
                    _success, camera_frame = camera.get_latest_frame()  # HWC
                    if camera_frame is None:
                        raise Exception("Camera frame is None")
                    processed_frame = CameraFrameProcessor.process(camera_frame)
                    state[camera_name] = processed_frame

                timestamp = time.perf_counter() - start_episode_t
                observation = self._build_geti_action_observation(state)
                if self.state.is_running:
                    if not self.action_queue:
                        self.action_queue = self.model.select_action(observation)[0].tolist()
                    action = self.action_queue.pop(0)

                    formatted_actions = dict(zip(self.action_keys, action))
                    await self.follower.set_joints_state(formatted_actions)
                    self._report_action(formatted_actions, state, timestamp)
                else:
                    self._report_action({}, state, timestamp)
                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / 30 - dt_s

                precise_sleep(wait_time)
        except Exception as e:
            traceback.print_exception(e)
            self._report_error(e)

    async def _on_start(self) -> None:
        logger.info("start")
        self.events["start"].clear()
        await self.follower.set_joints_state(SO_101_REST_POSITION)
        precise_sleep(0.3)  # TODO check if neccesary
        self.action_queue.clear()
        self.state.is_running = True
        self._report_state()

    async def _on_stop(self) -> None:
        logger.info("stop")
        self.events["stop"].clear()
        precise_sleep(0.3)  # TODO check if neccesary
        self.state.is_running = False
        self.action_queue.clear()
        self._report_state()

    async def teardown(self) -> None:
        """Disconnect robots and close queue."""
        if self.follower.is_connected:
            await self.follower.disconnect()

        # Wait for .5 seconds before closing queue to allow messages through
        await asyncio.sleep(0.5)
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

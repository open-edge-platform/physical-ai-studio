# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import time
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

import cv2
import numpy as np
from lerobot.utils.robot_utils import precise_sleep
from loguru import logger
from pydantic import BaseModel

from robots.robot_client import RobotClient
from robots.robot_client_factory import RobotClientFactory
from schemas import Model
from schemas.environment import EnvironmentWithRelations
from services.robot_calibration_service import RobotCalibrationService
from utils.serial_robot_tools import RobotConnectionManager
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration
from workers.inference.sync_mixed_model_integration import SyncMixedModelIntegration

from .base import BaseThreadWorker


class InferenceState(BaseModel):
    is_running: bool = False
    task_index: int = 0
    model_loaded: bool = False
    environment_loaded: bool = False
    error: bool = False  # TODO: Allow saving of state.


class InferenceWorker(BaseThreadWorker):
    ROLE: str = "InferenceWorker"

    robot_client_factory: RobotClientFactory

    queue: Queue
    state: InferenceState
    model_integration: SyncMixedModelIntegration | None = None
    environment_integration: InferenceEnvironmentIntegration | None = None
    fps: int = 30

    follower: RobotClient
    action_keys: list[str] = []
    camera_keys: list[str] = []

    events: dict[str, EventClass]

    def __init__(
        self,
        stop_event: EventClass,
        queue: Queue,
        calibration_service: RobotCalibrationService,
        robot_manager: RobotConnectionManager,
    ):
        super().__init__(stop_event=stop_event)
        self.queue = queue
        self.state = InferenceState()
        self.robot_client_factory = RobotClientFactory(robot_manager, calibration_service)
        self.events = {"interrupt": Event(), "new_model": Event(), "new_environment": Event()}

    def start_task(self, task_index: int) -> None:
        logger.info("start")
        if self.ready_for_inference:
            if self.model_integration is not None:
                self.model_integration.reset()
            self.state.is_running = True
            self.start_episode_t = time.perf_counter()
        self._report_state()

    def stop(self) -> None:
        """Stop inference."""
        logger.info("stop")
        self.state.is_running = False
        self.events["interrupt"].set()
        self._report_state()

    def disconnect(self) -> None:
        """Stop inference and teardown."""
        self.events["interrupt"].set()

    def load_model(self, model: Model, backend: str) -> None:
        # // TODO: Unload existing model worker
        try:
            self.model_integration = SyncMixedModelIntegration(
                model=model,
                backend=backend,
                stop_event=self._stop_event,
                fps=self.fps,
            )
            self.state.model_loaded = False
            self.events["new_model"].set()
            self._report_state()
        except Exception as e:
            self.model_integration = None
            self.state.error = True
            self._report_error(e)

    def load_environment(self, environment: EnvironmentWithRelations) -> None:
        """Setup environment."""

        try:
            self.environment_integration = InferenceEnvironmentIntegration(
                environment=environment, robot_client_factory=self.robot_client_factory
            )
            self.events["new_environment"].set()
            self.state.environment_loaded = False
            self._report_state()
        except Exception as e:
            self.environment_integration = None
            self.state.error = True
            self._report_error(e)

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        self._report_state()

    @property
    def ready_for_inference(self) -> bool:
        """Check if model and environment is loaded and no errors occurred."""
        return self.state.environment_loaded and self.state.model_loaded and not self.state.error

    async def run_loop(self) -> None:
        """inference loop."""
        try:
            logger.info("run loop")

            self.state.is_running = False
            self.start_episode_t = time.perf_counter()

            while not self.should_stop() and not self.events["interrupt"].is_set():
                if self.state.error:
                    return

                await self._handle_new_model_load()
                await self._handle_setup_environment()

                start_loop_t = time.perf_counter()
                if self.environment_integration:
                    observation = await self.environment_integration.get_observation()
                    timestamp = time.perf_counter() - self.start_episode_t
                    if observation:
                        report_observation = self.environment_integration.format_observation_for_reporting(observation, timestamp)
                        if self.state.is_running:
                            model_observation = self.environment_integration.format_model_input_observation(observation)
                            action = self.model_integration.select_action(model_observation)
                            if action is not None:
                                logger.info(action)
                                formatted_actions = dict(zip(self.environment_integration.action_keys, action))
                                report_observation["actions"] = formatted_actions
                                await self.environment_integration.set_joints_state(formatted_actions, 1 / 30)
                                # self._report_action(observation, formatted_actions, timestamp)
                        self._report_observation(report_observation)
                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / 30 - dt_s

                precise_sleep(wait_time)
        except Exception as e:
            logger.exception(f"Inference loop error: {e}")
            self._report_error(e)

    async def _handle_new_model_load(self) -> None:
        if self.model_integration and self.events["new_model"].is_set():
            self.events["new_model"].clear()
            await self.model_integration.setup()
            self.state.model_loaded = True
            self._report_state()

    async def _handle_setup_environment(self) -> None:
        if self.environment_integration and self.events["new_environment"].is_set():
            self.events["new_environment"].clear()
            await self.environment_integration.setup()
            self.state.environment_loaded = True
            self._report_state()

    async def teardown(self) -> None:
        """Disconnect robots and close queue."""
        if self.environment_integration:
            await self.environment_integration.teardown()

        if self.model_integration is not None:
            self.model_integration.teardown()

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

    def _report_observation(self, data: dict):
        """Report observation to queue."""
        self.queue.put(
            {
                "event": "observations",
                "data": data,
            }
        )

    def _base_64_encode_observation(self, observation: np.ndarray | None) -> str:
        if observation is None:
            return ""
        _, imagebytes = cv2.imencode(".jpg", observation)
        return base64.b64encode(imagebytes).decode()

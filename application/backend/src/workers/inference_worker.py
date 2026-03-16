# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from internal_datasets.mutations.recording_mutation import RecordingMutation
from internal_datasets.dataset_client import DatasetClient
import asyncio
import time
from pathlib import Path
from multiprocessing import Event, Queue
from multiprocessing.synchronize import Event as EventClass

from loguru import logger
from pydantic import BaseModel

from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from utils.dataset import build_lerobot_dataset_features
from robots.robot_client_factory import RobotClientFactory
from schemas import Model
from schemas.dataset import Episode, Dataset
from schemas.environment import EnvironmentWithRelations
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration
from workers.inference.sync_mixed_model_integration import SyncMixedModelIntegration

from .base import BaseThreadWorker


class InferenceState(BaseModel):
    is_running: bool = False
    task: str | None = None
    model_loaded: bool = False
    dataset_loaded: bool = False
    environment_loaded: bool = False
    is_recording: bool = False


class WorkerEvents:
    def __init__(self):
        self.interrupt = Event()
        self.new_model = Event()
        self.new_environment = Event()
        self.start_recording = Event()
        self.save_episode = Event()
        self.discard_episode = Event()
        self.start_recording_mutation = Event()



class InferenceWorker(BaseThreadWorker):
    ROLE: str = "InferenceWorker"

    robot_client_factory: RobotClientFactory

    queue: Queue
    state: InferenceState
    model_integration: SyncMixedModelIntegration | None = None
    environment_integration: InferenceEnvironmentIntegration | None = None
    dataset: DatasetClient | None = None
    recording_mutation: RecordingMutation | None = None

    fps: int = 30

    action_keys: list[str] = []
    camera_keys: list[str] = []

    events: WorkerEvents

    def __init__(
        self,
        stop_event: EventClass,
        queue: Queue,
        robot_client_factory: RobotClientFactory,
    ):
        super().__init__(stop_event=stop_event)
        self.queue = queue
        self.state = InferenceState()
        self.robot_client_factory = robot_client_factory
        self.events = WorkerEvents()

    def start_task(self, task: str) -> None:
        if self.ready_for_inference:
            if self.model_integration is not None:
                self.model_integration.reset()
            self.state.is_running = True
            self.state.task = task
            self.start_episode_t = time.perf_counter()
        self._report_state()

    def load_dataset(self, dataset: Dataset) -> None:
        self.dataset = InternalLeRobotDataset(Path(dataset.path))
        self.events.start_recording_mutation.set()

    def start_recording(self, task: str) -> None:
        self.state.task = task
        self.events.start_recording.set()

    def save_episode(self) -> None:
        self.events.save_episode.set()

    def discard_episode(self) -> None:
        self.events.discard_episode.set()

    def stop(self) -> None:
        """Stop inference."""
        self.state.is_running = False
        self._report_state()

    def disconnect(self) -> None:
        """Stop inference and teardown."""
        self.events.interrupt.set()

    def load_model(self, model: Model, backend: str) -> None:
        try:
            self.model_integration = SyncMixedModelIntegration(
                model=model,
                backend=backend,
                stop_event=self._stop_event,
                fps=self.fps,
            )
            self.state.model_loaded = False
            self.events.new_model.set()
            self._report_state()
        except Exception as e:
            self.model_integration = None
            self._report_error(e)

    def load_environment(self, environment: EnvironmentWithRelations) -> None:
        """Setup environment."""
        try:
            self.environment_integration = InferenceEnvironmentIntegration(
                environment=environment, robot_client_factory=self.robot_client_factory
            )
            self.events.new_environment.set()
            self.state.environment_loaded = False
            self._report_state()
        except Exception as e:
            self.environment_integration = None
            self._report_error(e)

    def setup(self) -> None:
        """Set up robots, cameras and dataset."""
        self._report_state()

    @property
    def ready_for_inference(self) -> bool:
        """Check if model and environment is loaded and no errors occurred."""
        return self.state.environment_loaded and self.state.model_loaded and self.state.task is not None

    @property
    def ready_for_recording(self) -> bool:
        """Check if model and environment is loaded and no errors occurred."""
        return self.state.environment_loaded and self.recording_mutation is not None and self.state.task is not None

    async def run_loop(self) -> None:
        """inference loop."""
        try:
            self.state.is_running = False
            self.start_episode_t = time.perf_counter()

            while not self.should_stop() and not self.events.interrupt.is_set():
                await asyncio.gather(
                    self._handle_new_model_load(),
                    self._handle_setup_environment(),
                    self._handle_start_mutation(),
                    self._handle_start_recording(),
                    self._handle_save_episode(),
                    self._handle_discard_episode(),
                )

                start_loop_t = time.perf_counter()
                if self.environment_integration:
                    actions = await self.environment_integration.set_follower_position_from_leader(1 / self.fps)
                    observation = await self.environment_integration.get_observation()
                    timestamp = time.perf_counter() - self.start_episode_t
                    if observation:
                        report_observation = self.environment_integration.format_observation_for_reporting(
                            observation, timestamp
                        )
                        if self.state.is_running and self.model_integration:
                            dataset_observation = self.environment_integration.format_model_input_observation(
                                observation, task=self.state.task
                            )
                            action = self.model_integration.select_action(dataset_observation)
                            if action is not None:
                                formatted_actions = dict(zip(self.environment_integration.action_keys, action))
                                report_observation["actions"] = formatted_actions
                                await self.environment_integration.set_joints_state(formatted_actions, 1 / 30)

                        if self.state.is_recording and self.ready_for_recording and actions:
                            dataset_observation = self.environment_integration.format_observation_for_dataset(observation)
                            logger.info(f"adding frame: {dataset_observation} {actions} {self.state.task}")
                            self.recording_mutation.add_frame(dataset_observation, actions, self.state.task)
                        self._report_observation(report_observation)
                dt_s = time.perf_counter() - start_loop_t
                wait_time = 1 / 30 - dt_s

                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.exception(f"Inference loop error: {e}")
            self._report_error(e)

    async def _handle_new_model_load(self) -> None:
        if self.model_integration and self.events.new_model.is_set():
            self.events.new_model.clear()
            await self.model_integration.setup()
            self.state.model_loaded = True
            logger.info("reporting state from new_model")
            self._report_state()

    async def _handle_setup_environment(self) -> None:
        if self.environment_integration and self.events.new_environment.is_set():
            self.events.new_environment.clear()
            await self.environment_integration.setup()
            self.state.environment_loaded = True
            logger.info("reporting state from setup_environment")
            self._report_state()

    async def _handle_start_recording(self) -> None:
        if self.ready_for_recording and self.events.start_recording.is_set():
            self.events.start_recording.clear()
            self.state.is_recording = True
            self._report_state()

    async def _handle_save_episode(self) -> None:
        if self.recording_mutation is not None and self.events.save_episode.is_set():
            self.events.save_episode.clear()
            self.recording_mutation.save_episode()
            self.state.is_recording = False
            self._report_state()

    async def _handle_discard_episode(self) -> None:
        if self.recording_mutation is not None and self.events.discard_episode.is_set():
            self.events.discard_episode.clear()
            self.recording_mutation.discard_buffer()
            self.state.is_recording = False
            self._report_state()

    async def _handle_start_mutation(self):
        if self.dataset and self.environment_integration and self.events.start_recording_mutation.is_set():
            self.events.start_recording_mutation.clear()
            features = await build_lerobot_dataset_features(self.environment_integration.environment, self.robot_client_factory)

            self.recording_mutation = self.dataset.start_recording_mutation(
                fps=self.fps,
                features=features,
                robot_type=self.environment_integration.follower.name,
            )
            self.state.dataset_loaded = True
            self._report_state()

    async def teardown(self) -> None:
        """Disconnect robots and close queue."""
        if self.environment_integration:
            await self.environment_integration.teardown()

        if self.model_integration is not None:
            self.model_integration.teardown()

        if self.recording_mutation:
            self.recording_mutation.teardown()

        # Wait for .5 seconds before closing queue to allow messages through
        await asyncio.sleep(0.5)
        self.queue.close()
        self.queue.cancel_join_thread()

    def _report_state(self):
        state = {"event": "state", "data": self.state.model_dump()}
        self.queue.put(state)

    def _report_error(self, error: BaseException):
        data = {
            "event": "error",
            "data": str(error),
        }
        logger.error(f"error: {data}")
        self.queue.put(data)

    def _report_observation(self, data: dict):
        """Report observation to queue."""
        self.queue.put(
            {
                "event": "observations",
                "data": data,
            }
        )

    def _report_episode(self, episode: Episode):
        self.queue.put(
            {
                "event": "episode",
                "data": episode.model_dump(),
            }
        )

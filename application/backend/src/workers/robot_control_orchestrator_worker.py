

from typing import Literal
from schemas.dataset import Dataset
from workers.recording_worker import RecordingWorker
from typing import Callable
from typing import Coroutine
from schemas import InferenceDevice, Model

from control.utils import get_observation_from_manifest, format_observation_for_reporting
from workers.model_integration import ModelIntegration
from workers.base import run_at_frequency
from typing import Any
from pydantic import BaseModel
from control.environment_integration import EnvironmentIntegration
from schemas.environment import EnvironmentWithRelations
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from robots.robot_client_factory import RobotClientFactory
import asyncio
from workers.base import BaseThreadWorker
from loguru import logger

class RobotControlState(BaseModel):
    task: str | None = None
    model_loaded: bool = False
    dataset_loaded: bool = False
    environment_loaded: bool = False
    is_recording: bool = False
    episodes_recorded: int = 0
    follower_source: Literal["model", "teleoperation"] | None = None

MESSAGE_QUEUE_FREQUENCY = 10


class RobotControlOrchestrator(BaseThreadWorker):
    ROLE = "RobotControlOrchestrator"

    recording: RecordingWorker | None = None
    model_integration: ModelIntegration | None = None
    environment_integration: EnvironmentIntegration | None = None
    background_tasks: set[asyncio.Task]

    def __init__(
        self, message_queue: asyncio.Queue, robot_client_factory: RobotClientFactory, mp_terminate_event: EventClass
    ):
        super().__init__(stop_event=mp_terminate_event)
        self.background_tasks = set()
        self.event_queue = mp.Queue()
        self._mp_terminate_event = mp_terminate_event
        self.robot_client_factory = robot_client_factory
        self.message_queue = message_queue
        self.state = RobotControlState()

    async def run_loop(self) -> None:
        while not self.should_stop():
            async with run_at_frequency(MESSAGE_QUEUE_FREQUENCY):
                while not self.event_queue.empty():
                    self._handle_event(self.event_queue.get())

    def _handle_event(self, event: dict) -> None:
        if event["event"] == "start_recording":
            self.state.is_recording = event["state"]["is_recording"]
            self._report_state()
        if event["event"] == "save_episode":
            self.state.is_recording = event["state"]["is_recording"]
            self.state.episodes_recorded = event["state"]["episodes_recorded"]
            self._report_state()
        if event["event"] == "discard_episode":
            self.state.is_recording = event["state"]["is_recording"]
            self._report_state()
        if event["event"] == "start_task":
            self.state.follower_source = "model" if event["state"]["is_running"] else "teleoperation"
            self._report_state()
        if event["event"] == "stop_task":
            self.state.follower_source = "model" if event["state"]["is_running"] else "teleoperation"
            self._report_state()

    def start_recording(self, task: str) -> None:
        """Start recording of specified task."""
        if self.recording:
            self.recording.start_episode(task)

    def save_episode(self) -> None:
        """Save recording."""
        if self.recording:
            self.recording.save_episode()

    def discard_episode(self) -> None:
        """Discard episode."""
        if self.recording:
            self.recording.discard_episode()

    def start_task(self, task: str) -> None:
        """Start task on model."""
        if self.model_integration:
            self.model_integration.start_task(task)
            self.set_follower_source("model")
    def stop_task(self) -> None:
        """Start task on model."""
        if self.model_integration:
            self.model_integration.stop_task()
            self.set_follower_source(None)

    def get_observation(self) -> dict | None:
        if self.environment_integration and self.environment_integration.manifest:
            obs = get_observation_from_manifest(self.environment_integration.manifest)
            return format_observation_for_reporting(obs, self.environment_integration.manifest)
        return None

    def set_follower_source(self, follower_source: Literal["model", "teleoperation"] | None) -> None:
        """Sets teleoperation loop to follow either model or teleoperator."""
        if self.environment_integration and self.environment_integration.manifest:
            action_source = 0
            if follower_source == "teleoperation":
                action_source = 1
            if follower_source == "model":
                action_source = 2
            self.environment_integration.manifest.robot.action_read_state.value = action_source
            self.state.follower_source = follower_source
            self._report_state()

    def load_model(self, model: Model, inference_device: InferenceDevice) -> None:
        if self.environment_integration and self.environment_integration.manifest:
            self.model_integration = ModelIntegration(
                model=model,
                inference_device=inference_device,
                data_manifest=self.environment_integration.manifest,
                mp_terminate_event=self._mp_terminate_event,
                event_queue=self.event_queue,
            )
            self.model_integration.start()
            self.fire_and_track(asyncio.to_thread(self.model_integration.loaded_event.wait), self._on_model_load)
        else:
            self._report_update("model_error", "cannot load model without environment")


    def _on_model_load(self, task: asyncio.Task) -> None:
        if task.cancelled():
            self.model_integration = None
            return
        exc = task.exception()
        if exc:
            self.model_integration = None
            logger.error(f"task failed: {exc}")
            self._report_update("model_error", str(exc))
        else:
            result = task.result()
            self.state.model_loaded = True
            self._report_update("model_loaded", result)
            self._report_state()

    def load_dataset(self, dataset: Dataset) -> None:
        """Load dataset and setup recording."""
        if self.environment_integration and self.environment_integration.manifest:
            worker = RecordingWorker(
                dataset=dataset,
                data_manifest=self.environment_integration.manifest,
                mp_terminate_event=self._mp_terminate_event,
                event_queue=self.event_queue,
            )
            worker.start()
            self.recording = worker
            self.fire_and_track(asyncio.to_thread(worker.loaded_event.wait), self._on_dataset_loaded)
        else:
            self._report_update("dataset_error", "cannot load dataset without environment")

    def _on_dataset_loaded(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"task failed: {exc}")
            self._report_update("dataset_error", str(exc))
        else:
            result = task.result()
            self.state.dataset_loaded = True
            self._report_update("dataset_loaded", result)
            self._report_state()



    def load_environment(self, environment: EnvironmentWithRelations) -> None:
        self.environment_integration = EnvironmentIntegration(
            environment=environment,
            robot_client_factory=self.robot_client_factory,
            mp_terminate_event=self._mp_terminate_event,
        )
        self.fire_and_track(self.environment_integration.setup_environment(), self._on_environment_loaded)

    def _on_environment_loaded(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"task failed: {exc}")
            self._report_update("environment_error", str(exc))
        else:
            result = task.result()
            self.state.environment_loaded = True
            self._report_update("environment_loaded", result)
            self._report_state()

    def fire_and_track(self, coro: Coroutine, on_done: Callable[[asyncio.Task],None])-> None:
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(on_done)
        task.add_done_callback(self.background_tasks.discard)


    def _report_update(self, event: str, message: Any)-> None:
        self.message_queue.put_nowait(
            {
                "event": event,
                "data": message,
            }
        )

    def _report_state(self):
        self.message_queue.put_nowait(
            {
                "event": "state",
                "data": self.state.model_dump(),
            }
        )

    async def teardown(self) -> None:
        if self.environment_integration:
            self.environment_integration.teardown()
        if self.recording:
            self.recording.stop()
        if self.model_integration:
            self.model_integration.stop()

        self.event_queue.close()

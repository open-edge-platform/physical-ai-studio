import asyncio
import time
from multiprocessing import Event, Queue
from unittest.mock import patch

import pytest

from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations
from services.robot_calibration_service import RobotCalibrationService
from settings import get_settings
from utils.serial_robot_tools import RobotConnectionManager
from workers.inference_worker import InferenceWorker

from .fixtures import test_environment


def get_next_item_from_queue_of_type(queue: Queue, event: str) -> dict | None:
    while not queue.empty():
        item = queue.get()
        print(item)
        if item["event"] == event:
            return item

    return None


class ControllableEnvironmentIntegration:
    observation: dict | None = None

    def __init__(self):
        self._setup_gate = Event()

    async def setup(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._setup_gate.wait)

    def allow_setup(self) -> None:
        self._setup_gate.set()

    async def get_observation(self):
        return self.observation

    def format_observation_for_reporting(self, obs, ts):
        return obs

    async def teardown(self):
        pass


@pytest.fixture
def inference_worker():
    stop_event = Event()
    robot_manager = RobotConnectionManager()
    settings = get_settings()
    calibration_service = RobotCalibrationService(robot_manager, settings)
    queue = Queue()

    process = InferenceWorker(
        stop_event=stop_event,
        robot_client_factory=RobotClientFactory(
            robot_manager=robot_manager,
            calibration_service=calibration_service,
        ),
        queue=queue,
    )
    process.start()

    yield process

    process.disconnect()
    process.join(timeout=5)


class TestInferenceWorker:
    def test_initialize(self, inference_worker: InferenceWorker):
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert report["data"] == {
            "is_running": False,
            "task": None,
            "model_loaded": False,
            "environment_loaded": False,
            "error": False,
        }

    def test_load_environment(self, inference_worker: InferenceWorker):
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        env_mock = ControllableEnvironmentIntegration()

        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=env_mock):
            inference_worker.load_environment(environment)
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert not report["data"]["environment_loaded"]

        env_mock.allow_setup()
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert report["data"]["environment_loaded"]

    def test_get_observations_once_environment_loaded(self, inference_worker: InferenceWorker):
        env_mock = ControllableEnvironmentIntegration()
        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=env_mock):
            inference_worker.load_environment(environment)
        env_mock.allow_setup()
        env_mock.observation = {"foo": "bar"}
        time.sleep(0.01)  # let feeder thread flush to pipe
        observation = get_next_item_from_queue_of_type(inference_worker.queue, "observations")
        assert observation is not None
        assert observation["event"] == "observations"
        assert observation["data"] == {"foo": "bar"}

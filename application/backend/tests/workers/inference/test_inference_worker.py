from workers.inference.sync_mixed_model_integration import SyncMixedModelIntegration
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration
import asyncio
import time
from multiprocessing import Event, Queue
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from robots.robot_client_factory import RobotClientFactory
from schemas.environment import EnvironmentWithRelations
from schemas.model import Model
from services.robot_calibration_service import RobotCalibrationService
from settings import get_settings
from utils.serial_robot_tools import RobotConnectionManager
from workers.inference_worker import InferenceWorker

from .fixtures import test_environment, test_observation


def wait_until_message_from_queue(queue: Queue, event: str, timeout: float=1):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        item = get_next_item_from_queue_of_type(queue, event)
        if item is not None:
            return item

        thread_flush()

    raise TimeoutError(f"No message in queue of event type: {event}")


def get_next_item_from_queue_of_type(queue: Queue, event: str) -> dict | None:
    while not queue.empty():
        item = queue.get()
        if item["event"] == event:
            return item

    return None

def clear_queue(queue: Queue) -> None:
    while not queue.empty():
        queue.get()

def thread_flush():
    """Small sleep to allow thread to work thru."""
    time.sleep(0.01)

@pytest.fixture
def model_integration():
    mock = MagicMock(spec=SyncMixedModelIntegration)

    gate = Event()

    async def controlled_setup():
        await asyncio.get_event_loop().run_in_executor(None, gate.wait)

    mock.setup = controlled_setup
    mock.allow_setup = gate.set
    mock.teardown = MagicMock()
    mock.select_action = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return mock


@pytest.fixture
def environment_integration():
    mock = MagicMock(spec=InferenceEnvironmentIntegration)

    gate = Event()

    async def controlled_setup():
        await asyncio.get_event_loop().run_in_executor(None, gate.wait)

    mock.setup = controlled_setup
    mock.allow_setup = gate.set
    mock.teardown = AsyncMock()
    mock.get_observation = AsyncMock(return_value=None)
    mock.format_observation_for_reporting = lambda obs, ts: obs
    mock.format_model_input_observation = lambda obs, task: obs

    return mock


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

@pytest.fixture
def loaded_inference_worker(inference_worker, environment_integration, model_integration):

    model = Model.model_validate(
        {
            "name": "foo",
            "policy": "act",
            "path": "/dev/null",
            "project_id": "35b48dc9-31df-40be-b295-08ae1d5378b1",
            "dataset_id": "93cffdc2-db6d-47bf-ac0c-4e5a727cbf0d",
            "properties": {},
            "snapshot_id": "f5e2cb67-3df2-4f16-bdfd-8b0782dd9e02",
        }
    )

    with patch("workers.inference_worker.SyncMixedModelIntegration", return_value=model_integration):
        inference_worker.load_model(model, "torch")

    environment = EnvironmentWithRelations.model_validate(test_environment)
    with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
        inference_worker.load_environment(environment)
    model_integration.allow_setup()
    environment_integration.allow_setup()
    thread_flush()
    clear_queue(inference_worker.queue)

    state = inference_worker.state
    assert state is not None
    assert state.model_loaded
    assert state.environment_loaded

    return inference_worker

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

    def test_load_environment(self, inference_worker: InferenceWorker, environment_integration):
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
            inference_worker.load_environment(environment)
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert not report["data"]["environment_loaded"]

        environment_integration.allow_setup()
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert report["data"]["environment_loaded"]

    def test_get_observations_once_environment_loaded(self, inference_worker: InferenceWorker, environment_integration):
        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
            inference_worker.load_environment(environment)
        environment_integration.allow_setup()
        environment_integration.get_observation = AsyncMock(return_value={"foo": "bar"})
        observation = wait_until_message_from_queue(inference_worker.queue, "observations")
        assert observation is not None
        assert observation["event"] == "observations"
        assert observation["data"] == {"foo": "bar"}

    def test_load_model(self, inference_worker: InferenceWorker, model_integration):
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        model = Model.model_validate(
            {
                "name": "foo",
                "policy": "act",
                "path": "/dev/null",
                "project_id": "35b48dc9-31df-40be-b295-08ae1d5378b1",
                "dataset_id": "93cffdc2-db6d-47bf-ac0c-4e5a727cbf0d",
                "properties": {},
                "snapshot_id": "f5e2cb67-3df2-4f16-bdfd-8b0782dd9e02",
            }
        )

        with patch("workers.inference_worker.SyncMixedModelIntegration", return_value=model_integration):
            inference_worker.load_model(model, "torch")
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert not report["data"]["model_loaded"]

        model_integration.allow_setup()
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert report["data"]["model_loaded"]

    def test_model_are_requested_with_actions(self,
                                              loaded_inference_worker: InferenceWorker,
                                              environment_integration,
                                              model_integration):
        worker = loaded_inference_worker
        worker.start_task("foo")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report is not None
        assert report["data"]["is_running"]
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_called_with(test_observation)

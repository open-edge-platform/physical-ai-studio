import asyncio
import multiprocessing as mp
from multiprocessing import Event
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.queue_utils import clear_queue, wait_until_message_from_queue

from control.environment_integration import EnvironmentIntegration
from workers.model_integration_worker import ModelIntegration
from workers.recording_worker import RecordingWorker
from workers.robot_control_orchestrator_worker import RobotControlOrchestrator


@pytest.fixture
def model_integration():
    mock = MagicMock(spec=ModelIntegration)
    mock.loaded_event = mp.Event()
    mock.loaded_event.set()
    return mock


@pytest.fixture
def environment_integration():
    mock = MagicMock(spec=EnvironmentIntegration)
    mock.setup_environment = AsyncMock()
    mock.teardown = MagicMock()
    mock.manifest = MagicMock()
    return mock


@pytest.fixture
def recording_worker():
    mock = MagicMock(spec=RecordingWorker)
    mock.loaded_event = mp.Event()
    mock.loaded_event.set()
    return mock


@pytest.fixture
def robot_control_worker(mock_robot_client_factory):
    stop_event = Event()
    queue = mp.Queue()

    process = RobotControlOrchestrator(
        message_queue=queue,
        robot_client_factory=mock_robot_client_factory,
        mp_terminate_event=stop_event,
    )
    process.start()

    yield process

    process.stop()
    process.join(timeout=5)


@pytest.fixture
def loaded_environment_worker(robot_control_worker, environment_integration, test_environment):
    with patch(
        "workers.robot_control_orchestrator_worker.EnvironmentIntegration", return_value=environment_integration
    ):
        asyncio.run(robot_control_worker.load_environment(test_environment))

    state = wait_until_message_from_queue(robot_control_worker.message_queue, "state")
    assert state["data"]["environment_loaded"]
    clear_queue(robot_control_worker.message_queue)

    return robot_control_worker


@pytest.fixture
def loaded_inference_worker(loaded_environment_worker, model_integration, test_model):
    worker = loaded_environment_worker
    with patch("workers.robot_control_orchestrator_worker.ModelIntegration", return_value=model_integration):
        asyncio.run(worker.load_model(test_model, "torch"))

    state = wait_until_message_from_queue(worker.message_queue, "state")
    assert state["data"]["model_loaded"]
    clear_queue(worker.message_queue)

    return worker


@pytest.fixture
def loaded_teleoperation_worker(loaded_environment_worker, recording_worker, test_dataset):
    worker = loaded_environment_worker
    with patch("workers.robot_control_orchestrator_worker.RecordingWorker", return_value=recording_worker):
        asyncio.run(worker.load_dataset(test_dataset))

    state = wait_until_message_from_queue(worker.message_queue, "state")
    assert state["data"]["dataset_loaded"]
    clear_queue(worker.message_queue)

    return worker


class TestRobotControlOrchestrator:
    def test_initialize(self, robot_control_worker: RobotControlOrchestrator):
        assert robot_control_worker.state.task is None
        assert not robot_control_worker.state.model_loaded
        assert not robot_control_worker.state.environment_loaded
        assert not robot_control_worker.state.is_recording
        assert not robot_control_worker.state.dataset_loaded
        assert robot_control_worker.state.follower_source is None
        assert robot_control_worker.state.episodes_recorded == 0

    def test_load_environment(self, robot_control_worker, environment_integration, test_environment):
        with patch(
            "workers.robot_control_orchestrator_worker.EnvironmentIntegration", return_value=environment_integration
        ):
            asyncio.run(robot_control_worker.load_environment(test_environment))

        state = wait_until_message_from_queue(robot_control_worker.message_queue, "state")
        assert state["data"]["environment_loaded"]

    def test_load_model(self, loaded_environment_worker, model_integration, test_model):
        worker = loaded_environment_worker
        with patch("workers.robot_control_orchestrator_worker.ModelIntegration", return_value=model_integration):
            asyncio.run(worker.load_model(test_model, "torch"))

        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert state["data"]["model_loaded"]

    def test_load_dataset(self, loaded_environment_worker, recording_worker, test_dataset):
        worker = loaded_environment_worker
        with patch("workers.robot_control_orchestrator_worker.RecordingWorker", return_value=recording_worker):
            asyncio.run(worker.load_dataset(test_dataset))

        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert state["data"]["dataset_loaded"]

    def test_starting_task_sets_follower_source_to_model(self, loaded_inference_worker):
        worker = loaded_inference_worker
        worker.start_task("foo")
        assert worker.state.follower_source == "model"

    def test_stop_task_resets_follower_source(self, loaded_inference_worker):
        worker = loaded_inference_worker
        worker.start_task("foo")
        worker.stop_task()
        assert worker.state.follower_source is None

    def test_set_follower_source(self, loaded_inference_worker):
        worker = loaded_inference_worker
        worker.set_follower_source("teleoperation")
        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert state["data"]["follower_source"] == "teleoperation"

        clear_queue(worker.message_queue)
        worker.set_follower_source(None)
        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert state["data"]["follower_source"] is None

    def test_teardown_calls_sub_worker_teardown(
        self, loaded_inference_worker, environment_integration, model_integration
    ):
        worker = loaded_inference_worker
        worker.stop()
        environment_integration.teardown.assert_called()
        model_integration.stop.assert_called()

    def test_recording_state_updates_from_events(self, loaded_teleoperation_worker, recording_worker):
        worker = loaded_teleoperation_worker
        worker.start_recording("task")
        recording_worker.start_episode.assert_called_with("task")

        worker.event_queue.put({"event": "start_recording", "state": {"is_recording": True}})
        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert state["data"]["is_recording"]

        worker.save_episode()
        recording_worker.save_episode.assert_called()

        worker.event_queue.put({"event": "save_episode", "state": {"is_recording": False, "episodes_recorded": 1}})
        state = wait_until_message_from_queue(worker.message_queue, "state")
        assert not state["data"]["is_recording"]
        assert state["data"]["episodes_recorded"] == 1

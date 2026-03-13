import asyncio
from multiprocessing import Event, Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.environment import EnvironmentWithRelations
from workers.inference.inference_environment_integration import InferenceEnvironmentIntegration
from workers.inference.sync_mixed_model_integration import SyncMixedModelIntegration
from workers.inference_worker import InferenceWorker

from .queue_utils import clear_queue, thread_flush, wait_until_message_from_queue


@pytest.fixture
def model_integration():
    mock = MagicMock(spec=SyncMixedModelIntegration)

    gate = Event()

    async def controlled_setup():
        await asyncio.get_event_loop().run_in_executor(None, gate.wait)

    mock.setup = controlled_setup
    mock.allow_setup = gate.set
    mock.teardown = MagicMock()
    mock.select_action = MagicMock(
        return_value=[
            -11.076923076923077,
            56.043956043956044,
            -10.197802197802197,
            69.45054945054945,
            -24.791208791208792,
            12.364425162689804,
        ]
    )
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
def inference_worker(mock_robot_client_factory):
    stop_event = Event()
    queue = Queue()

    process = InferenceWorker(
        stop_event=stop_event,
        robot_client_factory=mock_robot_client_factory,
        queue=queue,
    )
    process.start()

    yield process

    process.disconnect()
    process.join(timeout=5)


@pytest.fixture
def loaded_inference_worker(inference_worker, environment_integration, model_integration, test_model, test_environment):
    with patch("workers.inference_worker.SyncMixedModelIntegration", return_value=model_integration):
        inference_worker.load_model(test_model, "torch")

    with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
        inference_worker.load_environment(test_environment)
    model_integration.allow_setup()
    thread_flush()
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
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"
        assert report["data"] == {
            "is_running": False,
            "task": None,
            "model_loaded": False,
            "environment_loaded": False,
            "error": False,
        }

    def test_load_environment(self, inference_worker: InferenceWorker, environment_integration, test_environment):
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"
        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
            inference_worker.load_environment(environment)
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"
        assert not report["data"]["environment_loaded"]

        environment_integration.allow_setup()
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"
        assert report["data"]["environment_loaded"]

    def test_get_observations_once_environment_loaded(
        self, inference_worker: InferenceWorker, environment_integration, test_environment
    ):
        environment_integration.get_observation = AsyncMock(return_value={"foo": "bar"})

        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.inference_worker.InferenceEnvironmentIntegration", return_value=environment_integration):
            inference_worker.load_environment(environment)
        environment_integration.allow_setup()
        observation = wait_until_message_from_queue(inference_worker.queue, "observations")
        assert observation is not None
        assert observation["event"] == "observations"
        assert observation["data"] == {"foo": "bar"}

    def test_load_model(self, inference_worker: InferenceWorker, model_integration, test_model):
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"

        with patch("workers.inference_worker.SyncMixedModelIntegration", return_value=model_integration):
            inference_worker.load_model(test_model, "torch")
        report = wait_until_message_from_queue(inference_worker.queue, "state")
        assert report["event"] == "state"
        assert not report["data"]["model_loaded"]

        model_integration.allow_setup()
        report = inference_worker.queue.get()
        assert report["event"] == "state"
        assert report["data"]["model_loaded"]

    def test_model_are_requested_with_actions(
        self, loaded_inference_worker: InferenceWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.start_task("foo")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report is not None
        assert report["data"]["is_running"]
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_called_with(test_observation)

    def test_stop_causes_model_inference_to_not_be_called(
        self, loaded_inference_worker: InferenceWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.start_task("foo")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report["data"]["is_running"]
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        worker.stop()
        # clear existing queue and wait for next observation
        report = wait_until_message_from_queue(worker.queue, "state")
        assert not report["data"]["is_running"]
        clear_queue(worker.queue)
        model_integration.select_action.reset()  # Reset mock of model select action
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_not_called()

    def test_disconnect_causes_teardown(
        self, loaded_inference_worker: InferenceWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.disconnect()
        worker.join()

        model_integration.teardown.assert_called()
        environment_integration.teardown.assert_awaited_once()

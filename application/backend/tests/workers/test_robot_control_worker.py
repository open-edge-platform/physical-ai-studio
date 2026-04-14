import asyncio
import time
from multiprocessing import Event, Queue
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from tests.queue_utils import clear_queue, thread_flush, wait_until_message_from_queue

from control.environment_integration import EnvironmentIntegration
from control.sync_mixed_model_integration import SyncMixedModelIntegration
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.recording_mutation import RecordingMutation
from schemas.environment import EnvironmentWithRelations
from workers.robot_control_worker import RobotControlWorker


def _wait_until_state(queue: Queue, timeout: float = 2, **expected: bool) -> dict:
    """Drain 'state' messages until all *expected* fields are ``True``."""
    t = time.perf_counter()
    latest = None
    while time.perf_counter() - t < timeout:
        try:
            msg = wait_until_message_from_queue(queue, "state", timeout=0.2)
            latest = msg["data"]
            if all(latest.get(k) == v for k, v in expected.items()):
                return latest
        except TimeoutError:
            pass
    raise TimeoutError(f"State never reached {expected}; last seen: {latest}")


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
    mock = MagicMock(spec=EnvironmentIntegration)

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
def recording_mutation():
    mock = MagicMock(spec=RecordingMutation)
    mock.add_frame = MagicMock()
    return mock


@pytest.fixture
def test_dataset_impl(recording_mutation):
    mock = MagicMock(spec=InternalLeRobotDataset)
    mock.start_recording_mutation = MagicMock(return_value=recording_mutation)
    return mock


@pytest.fixture
def robot_control_worker(mock_robot_client_factory):
    stop_event = Event()
    queue = Queue()
    mock_registry = MagicMock()
    mock_registry.acquire = AsyncMock(return_value=(uuid4(), MagicMock()))
    mock_registry.release = AsyncMock()

    process = RobotControlWorker(
        stop_event=stop_event,
        robot_client_factory=mock_robot_client_factory,
        queue=queue,
        model_worker_registry=mock_registry,
    )
    process.start()

    yield process

    process.disconnect()
    process.join(timeout=5)


@pytest.fixture
def loaded_inference_worker(
    robot_control_worker, environment_integration, model_integration, test_model, test_environment
):
    with patch("workers.robot_control_worker.SyncMixedModelIntegration", return_value=model_integration):
        robot_control_worker.load_model(test_model, "torch")
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
            robot_control_worker.load_environment(test_environment)
        model_integration.allow_setup()
        environment_integration.allow_setup()
        state = _wait_until_state(robot_control_worker.queue, model_loaded=True, environment_loaded=True)

    assert state["model_loaded"]
    assert state["environment_loaded"]
    clear_queue(robot_control_worker.queue)

    return robot_control_worker


@pytest.fixture
def loaded_teleoperation_worker(
    robot_control_worker, environment_integration, test_dataset_impl, test_dataset, test_environment
):
    with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
        robot_control_worker.load_environment(test_environment)
    environment_integration.allow_setup()
    with patch("workers.robot_control_worker.InternalLeRobotDataset", return_value=test_dataset_impl):
        robot_control_worker.load_dataset(test_dataset)
    thread_flush()
    clear_queue(robot_control_worker.queue)
    state = wait_until_message_from_queue(robot_control_worker.queue, "state")["data"]
    assert state["environment_loaded"]
    assert state["dataset_loaded"]

    return robot_control_worker


class TestRobotControlWorker:
    def test_initialize(self, robot_control_worker: RobotControlWorker):
        report = wait_until_message_from_queue(robot_control_worker.queue, "state")
        assert report["event"] == "state"
        assert report["data"] == {
            "task": None,
            "model_loaded": False,
            "episodes_recorded": 0,
            "environment_loaded": False,
            "is_recording": False,
            "dataset_loaded": False,
            "follower_source": None,
        }

    def test_load_environment(
        self, robot_control_worker: RobotControlWorker, environment_integration, test_environment
    ):
        report = wait_until_message_from_queue(robot_control_worker.queue, "state")
        assert report["event"] == "state"
        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
            robot_control_worker.load_environment(environment)
        report = wait_until_message_from_queue(robot_control_worker.queue, "state")
        assert report["event"] == "state"
        assert not report["data"]["environment_loaded"]

        environment_integration.allow_setup()
        report = wait_until_message_from_queue(robot_control_worker.queue, "state")
        assert report["event"] == "state"
        assert report["data"]["environment_loaded"]

    def test_get_observations_once_environment_loaded(
        self, robot_control_worker: RobotControlWorker, environment_integration, test_environment
    ):
        environment_integration.get_observation = AsyncMock(return_value={"foo": "bar"})

        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
            robot_control_worker.load_environment(environment)
        environment_integration.allow_setup()
        observation = wait_until_message_from_queue(robot_control_worker.queue, "observations")
        assert observation is not None
        assert observation["event"] == "observations"
        assert observation["data"] == {"foo": "bar"}

    def test_load_model(self, robot_control_worker: RobotControlWorker, model_integration, test_model):
        report = wait_until_message_from_queue(robot_control_worker.queue, "state")
        assert report["event"] == "state"

        # Keep patch active until after allow_setup(): _handle_new_model_load creates
        # SyncMixedModelIntegration asynchronously, so the patch must outlive load_model().
        with patch("workers.robot_control_worker.SyncMixedModelIntegration", return_value=model_integration):
            robot_control_worker.load_model(test_model, "torch")
            report = wait_until_message_from_queue(robot_control_worker.queue, "state")
            assert report["event"] == "state"
            assert not report["data"]["model_loaded"]

            model_integration.allow_setup()
            report = robot_control_worker.queue.get()

        assert report["event"] == "state"
        assert report["data"]["model_loaded"]

    def test_model_are_requested_with_actions(
        self, loaded_inference_worker: RobotControlWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.start_task("foo")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report is not None
        assert report["data"]["follower_source"] == "model"
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_called_with(test_observation)

    def test_stop_causes_model_inference_to_not_be_called(
        self, loaded_inference_worker: RobotControlWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.start_task("foo")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report["data"]["follower_source"] == "model"
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        worker.stop()
        # clear existing queue and wait for next observation
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report["data"]["follower_source"] is None
        clear_queue(worker.queue)
        model_integration.select_action.reset_mock()  # Reset mock of model select action
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_not_called()

    def test_disconnect_causes_teardown(
        self, loaded_inference_worker: RobotControlWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.disconnect()
        worker.join()

        model_integration.teardown.assert_called()
        environment_integration.teardown.assert_awaited_once()

    def test_starting_task_sets_follower_source_to_model(
        self, loaded_inference_worker: RobotControlWorker, environment_integration, model_integration, test_observation
    ):
        worker = loaded_inference_worker
        worker.start_task("foo")
        assert worker.state.follower_source == "model"

    def test_select_follower_input(
        self, loaded_inference_worker: RobotControlWorker, environment_integration, model_integration, test_observation
    ):
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        worker = loaded_inference_worker
        worker.start_task("foo")  # start task automatically sets follower input to model
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_called_with(test_observation)
        worker.set_follower_source(None)
        wait_until_message_from_queue(worker.queue, "state")
        clear_queue(worker.queue)
        model_integration.select_action.reset_mock()  # Reset mock of model select action
        wait_until_message_from_queue(worker.queue, "observations")
        model_integration.select_action.assert_not_called()

    def test_teleoperation_recording(
        self,
        loaded_teleoperation_worker: RobotControlWorker,
        environment_integration,
        test_dataset,
        test_observation,
        recording_mutation,
        test_actions,
    ):
        """Tests the entire recording via teleoperation flow."""
        worker = loaded_teleoperation_worker
        worker.set_follower_source("teleoperation")
        report = wait_until_message_from_queue(worker.queue, "state")
        assert report is not None
        assert report["data"]["follower_source"] == "teleoperation"
        worker.start_recording("Foo bar")
        environment_integration.get_observation = AsyncMock(return_value=test_observation)
        environment_integration.set_follower_position_from_leader = AsyncMock(return_value=test_actions)
        observation = wait_until_message_from_queue(worker.queue, "observations")
        assert observation is not None
        recording_mutation.add_frame.assert_called()
        worker.save_episode()
        report = wait_until_message_from_queue(worker.queue, "state")
        assert not report["data"]["is_recording"]
        assert report["data"]["episodes_recorded"] == 1
        recording_mutation.save_episode.assert_called()
        worker.disconnect()
        worker.join()
        recording_mutation.teardown.assert_called()

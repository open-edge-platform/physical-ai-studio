import asyncio
import time
from multiprocessing import Event, Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.queue_utils import clear_queue, wait_until_message_from_queue

from control.environment_integration import EnvironmentIntegration
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

    process = RobotControlWorker(
        stop_event=stop_event,
        robot_client_factory=mock_robot_client_factory,
        queue=queue,
    )
    process.start()

    yield process

    process.disconnect()
    process.join(timeout=5)


@pytest.fixture
def loaded_teleoperation_worker(
    robot_control_worker, environment_integration, test_dataset_impl, test_dataset, test_environment
):
    with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
        robot_control_worker.load_environment(test_environment)
    environment_integration.allow_setup()
    with patch("workers.robot_control_worker.InternalLeRobotDataset", return_value=test_dataset_impl):
        robot_control_worker.load_dataset(test_dataset)
    state = _wait_until_state(robot_control_worker.queue, environment_loaded=True, dataset_loaded=True)
    assert state["environment_loaded"]
    assert state["dataset_loaded"]
    clear_queue(robot_control_worker.queue)

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

    def test_environment_setup_failure_reports_error_and_keeps_worker_alive(
        self, robot_control_worker: RobotControlWorker, test_environment
    ):
        failing_integration = MagicMock(spec=EnvironmentIntegration)
        failing_integration.setup = AsyncMock(side_effect=RuntimeError("robot connect failed"))
        failing_integration.teardown = AsyncMock()

        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=failing_integration):
            robot_control_worker.load_environment(environment)

        error_report = wait_until_message_from_queue(robot_control_worker.queue, "error")
        assert error_report["message"] == "robot connect failed"
        assert error_report["error_code"] == "robot_control_error"
        assert robot_control_worker.is_alive()

        succeeding_integration = MagicMock(spec=EnvironmentIntegration)
        succeeding_integration.setup = AsyncMock()
        succeeding_integration.get_observation = AsyncMock(return_value=None)

        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=succeeding_integration):
            robot_control_worker.load_environment(environment)

        state_report = _wait_until_state(robot_control_worker.queue, environment_loaded=True)
        assert state_report["environment_loaded"]
        assert robot_control_worker.is_alive()

    def test_environment_setup_reports_app_exception_fields(
        self, robot_control_worker: RobotControlWorker, test_environment
    ):
        from exceptions import RobotDeviceAlreadyOwnedError

        failing_integration = MagicMock(spec=EnvironmentIntegration)
        failing_integration.setup = AsyncMock(side_effect=RobotDeviceAlreadyOwnedError(device_ids=("serial:ttyACM0",)))
        failing_integration.teardown = AsyncMock()

        environment = EnvironmentWithRelations.model_validate(test_environment)
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=failing_integration):
            robot_control_worker.load_environment(environment)

        error_report = wait_until_message_from_queue(robot_control_worker.queue, "error")
        assert error_report["error_code"] == "robot_device_already_owned"
        assert "serial:ttyACM0" in error_report["message"]
        assert robot_control_worker.is_alive()

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

    def test_disconnect_causes_teardown(
        self, robot_control_worker: RobotControlWorker, environment_integration, test_environment
    ):
        with patch("workers.robot_control_worker.EnvironmentIntegration", return_value=environment_integration):
            robot_control_worker.load_environment(test_environment)
        environment_integration.allow_setup()
        _wait_until_state(robot_control_worker.queue, environment_loaded=True)

        robot_control_worker.disconnect()
        robot_control_worker.join()

        environment_integration.teardown.assert_awaited_once()

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

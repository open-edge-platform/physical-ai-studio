import multiprocessing as mp
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from robots.catalog.so101 import SO101Robot, SO101RobotPayload
from robots.robot_client_factory import RobotClientFactory
from schemas.robot import Robot
from workers.teleoperate_worker import ActionReadState, TeleoperateWorker

FEATURES = ["joint1", "joint2", "joint3"]
LEADER_ROBOT_TYPES = {
    "SO101_Leader",
    "Trossen_WidowXAI_Leader",
    "Trossen_Bimanual_WidowXAI_Leader",
}


def _make_client(state: dict | None = None):
    state = state or {k: float(i) for i, k in enumerate(FEATURES)}
    client = MagicMock()
    client.features.return_value = FEATURES
    client.read_state.return_value = {"state": state}
    return client


def _make_robot_schema(robot_id: str = "robot-1", robot_type: str = "SO101_Follower") -> Robot:
    return SO101Robot(
        id=uuid4(),
        name=f"Robot {robot_id}",
        type=robot_type,  # type: ignore[arg-type]
        payload=SO101RobotPayload(connection_string="/dev/null"),
    )


async def _make_factory(follower_client=None, leader_client=None):
    """Create a mock factory that returns configured clients."""
    factory = MagicMock(spec=RobotClientFactory)
    follower_client = follower_client or _make_client()

    async def build_side_effect(robot_schema):
        if leader_client is not None and robot_schema.type in LEADER_ROBOT_TYPES:
            return leader_client
        return follower_client

    factory.build = AsyncMock(side_effect=build_side_effect)
    return factory, follower_client, leader_client


@asynccontextmanager
async def _noop_frequency(*args, **kwargs):
    yield


class TestTeleoperateWorkerBuffers:
    """Test the thread-shared observation buffer."""

    def test_initial_state_is_empty(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        assert worker.get_state() == []

    def test_state_round_trip(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        worker._set_state([1.0, 2.0, 3.0])
        assert worker.get_state() == [1.0, 2.0, 3.0]

    def test_action_read_state_defaults_to_none(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        assert worker.get_action_read_state() == ActionReadState.NONE

    def test_set_action_read_state_with_enum(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        worker.set_action_read_state(ActionReadState.TELEOPERATION)
        assert worker.get_action_read_state() == ActionReadState.TELEOPERATION

    def test_set_action_read_state_with_int(self):
        """Validate and normalize input with ActionReadState()."""
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        worker.set_action_read_state(1)  # ActionReadState.TELEOPERATION
        assert worker.get_action_read_state() == ActionReadState.TELEOPERATION

        with pytest.raises(ValueError):
            worker.set_action_read_state(999)  # Invalid value


class TestTeleoperateWorkerRunLoop:
    """Test the async run_loop with factory-based client building."""

    async def test_builds_follower_and_leader_from_factory(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        leader_client = _make_client()
        factory, _, _ = await _make_factory(follower_client, leader_client)

        follower_schema = _make_robot_schema("follower-1")
        leader_schema = _make_robot_schema("leader-1", "SO101_Leader")

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=leader_schema,
            frequency=30.0,
            stop_event=stop_event,
        )

        # Set up to stop after one loop iteration
        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stop_event.set()
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = stop_after_read

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        # Verify factory was called for both
        assert factory.build.call_count == 2
        assert worker.follower is not None
        assert worker.leader is not None

    async def test_connects_robots_before_features(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        factory, _, _ = await _make_factory(follower_client)

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stop_event.set()
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = stop_after_read

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        # Verify connect was called before features
        follower_client.connect.assert_called_once()
        follower_client.features.assert_called_once()
        assert worker.features == FEATURES

    async def test_sets_loaded_event_after_successful_setup(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        factory, _, _ = await _make_factory(follower_client)

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        assert not worker.loaded_event.is_set()
        assert worker.setup_error is None

        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stop_event.set()
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = stop_after_read

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        assert worker.loaded_event.is_set()
        assert worker.setup_error is None

    async def test_stores_setup_error_and_doesnt_set_loaded_event(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        factory.build = AsyncMock(side_effect=RuntimeError("Build failed"))

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        assert not worker.loaded_event.is_set()
        assert worker.setup_error is not None
        assert isinstance(worker.setup_error, RuntimeError)
        assert "Build failed" in str(worker.setup_error)

    async def test_disconnects_all_clients_on_stop(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        leader_client = _make_client()
        factory, _, _ = await _make_factory(follower_client, leader_client)

        follower_schema = _make_robot_schema("follower-1")
        leader_schema = _make_robot_schema("leader-1", "SO101_Leader")

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=leader_schema,
            frequency=30.0,
            stop_event=stop_event,
        )

        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stop_event.set()
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = stop_after_read

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        follower_client.disconnect.assert_called_once()
        leader_client.disconnect.assert_called_once()

    async def test_disconnects_only_built_clients_on_setup_failure(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        factory = MagicMock(spec=RobotClientFactory)

        # First call succeeds for follower, second fails for leader
        factory.build = AsyncMock(side_effect=[follower_client, RuntimeError("Leader build failed")])

        follower_schema = _make_robot_schema("follower-1")
        leader_schema = _make_robot_schema("leader-1", "SO101_Leader")

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=leader_schema,
            frequency=30.0,
            stop_event=stop_event,
        )

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        assert not worker.loaded_event.is_set()
        assert worker.setup_error is not None
        # Follower should be disconnected despite setup failure
        follower_client.disconnect.assert_called_once()

    async def test_initial_state_populated_before_loaded_event(self):
        stop_event = mp.Event()
        follower_state = {"joint1": 1.1, "joint2": 2.2, "joint3": 3.3}
        follower_client = _make_client(state=follower_state)
        factory, _, _ = await _make_factory(follower_client)

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stop_event.set()
            return {"state": follower_state}

        follower_client.read_state.side_effect = stop_after_read

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        assert worker.get_state() == [1.1, 2.2, 3.3]
        assert worker.loaded_event.is_set()

    async def test_teleoperation_mode_sends_leader_to_follower(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        leader_state = {"joint1": 10.0, "joint2": 20.0, "joint3": 30.0}
        leader_client = _make_client(state=leader_state)
        factory, _, _ = await _make_factory(follower_client, leader_client)

        follower_schema = _make_robot_schema("follower-1")
        leader_schema = _make_robot_schema("leader-1", "SO101_Leader")

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=leader_schema,
            frequency=30.0,
            stop_event=stop_event,
        )

        worker.set_action_read_state(ActionReadState.TELEOPERATION)

        call_count = 0

        def stop_after_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop_event.set()
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = stop_after_read
        leader_client.read_state.return_value = {"state": leader_state}

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        # follower should have set_joints_state called with leader values
        follower_client.set_joints_state.assert_called()

    async def test_setup_failure_does_not_propagate(self):
        """The API caller is still waiting, so the reason is stored, not raised."""
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        factory.build = AsyncMock(side_effect=RuntimeError("Build failed"))

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=_make_robot_schema("follower-1"),
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            await worker.run_loop()

        assert isinstance(worker.setup_error, RuntimeError)
        assert not worker.loaded_event.is_set()

    async def test_failure_after_load_propagates_and_still_disconnects(self):
        """Once loaded, a fault is a runtime error for BaseThreadWorker.run to log."""
        stop_event = mp.Event()
        follower_client = _make_client()
        leader_client = _make_client()
        factory, _, _ = await _make_factory(follower_client, leader_client)

        call_count = 0

        def fail_on_second_read():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("Robot read failed mid-loop")
            return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

        follower_client.read_state.side_effect = fail_on_second_read

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=_make_robot_schema("follower-1"),
            leader=_make_robot_schema("leader-1", "SO101_Leader"),
            frequency=30.0,
            stop_event=stop_event,
        )

        with (
            patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency),
            pytest.raises(RuntimeError, match="mid-loop"),
        ):
            await worker.run_loop()

        # A loop fault must not be mistaken for a setup failure.
        assert worker.loaded_event.is_set()
        assert worker.setup_error is None
        follower_client.disconnect.assert_called_once()
        leader_client.disconnect.assert_called_once()


class TestWaitUntilLoaded:
    """Test the failure-aware readiness wait method."""

    async def test_wait_until_loaded_returns_when_ready(self):
        stop_event = mp.Event()
        follower_client = _make_client()
        factory, _, _ = await _make_factory(follower_client)

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        worker.loaded_event.set()

        # Should return immediately without error
        await worker.wait_until_loaded(poll_interval=0.001)

    async def test_wait_until_loaded_raises_setup_error(self):
        stop_event = mp.Event()
        factory = MagicMock(spec=RobotClientFactory)
        factory.build = AsyncMock(side_effect=RuntimeError("Setup failed"))

        follower_schema = _make_robot_schema()

        worker = TeleoperateWorker(
            robot_client_factory=factory,
            follower=follower_schema,
            leader=None,
            frequency=30.0,
            stop_event=stop_event,
        )

        # Simulate setup error being set
        worker.setup_error = RuntimeError("Setup failed")

        with pytest.raises(RuntimeError, match="Setup failed"):
            await worker.wait_until_loaded(poll_interval=0.001)

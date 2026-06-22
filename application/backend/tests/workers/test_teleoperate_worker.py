import asyncio
import multiprocessing as mp
from contextlib import asynccontextmanager
from multiprocessing.synchronize import Event as EventClass
from unittest.mock import MagicMock, patch

from workers.teleoperate_worker import ActionReadState, TeleoperateWorker

FEATURES = ["joint1", "joint2", "joint3"]


def _make_client(state: dict | None = None):
    state = state or {k: float(i) for i, k in enumerate(FEATURES)}
    client = MagicMock()
    client.features.return_value = FEATURES
    client.read_state.return_value = {"state": state}
    return client


def _make_worker(follower=None, leader=None, frequency=30.0, stop_event=None):
    follower = follower or _make_client()
    stop_event = stop_event or mp.Event()
    worker = TeleoperateWorker(follower=follower, leader=leader, frequency=frequency, mp_stop_event=stop_event)
    asyncio.run(worker.setup())
    return worker


@asynccontextmanager
async def _noop_frequency(*args, **kwargs):
    yield


def _stop_after(stop_event: EventClass, n: int):
    """Returns a read_state side effect that sets stop_event after n calls."""
    call_count = 0

    def read_state():
        nonlocal call_count
        call_count += 1
        if call_count >= n:
            stop_event.set()
        return {"state": {k: float(i) for i, k in enumerate(FEATURES)}}

    return read_state


class TestTeleoperateWorkerSharedMemory:
    def test_state_is_zeros_initially(self):
        worker = _make_worker()
        assert worker.get_state() == [0.0] * len(FEATURES)

    def test_state_round_trip(self):
        worker = _make_worker()
        worker._set_state([1.0, 2.0, 3.0])
        assert worker.get_state() == [1.0, 2.0, 3.0]

    def test_actions_are_follower_state_initially(self):
        worker = _make_worker()
        assert worker.get_actions() == [0.0, 1.0, 2.0]

    def test_actions_round_trip(self):
        worker = _make_worker()
        worker._set_actions([4.0, 5.0, 6.0])
        assert worker.get_actions() == [4.0, 5.0, 6.0]

    def test_action_source_defaults_to_none(self):
        worker = _make_worker()
        assert worker.get_action_read_state() == ActionReadState.NONE

    def test_set_action_read_state(self):
        worker = _make_worker()
        worker.set_action_read_state(ActionReadState.TELEOPERATION)
        assert worker.get_action_read_state() == ActionReadState.TELEOPERATION
        worker.set_action_read_state(ActionReadState.FROM_ACTIONS)
        assert worker.get_action_read_state() == ActionReadState.FROM_ACTIONS


class TestTeleoperateWorkerRunLoop:
    def test_connects_follower_and_disconnects_on_stop(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)

        worker = _make_worker(follower=follower, stop_event=stop_event)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        follower.connect.assert_called_once()
        follower.disconnect.assert_called_once()

    def test_connects_and_disconnects_leader(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)
        leader = _make_client()

        worker = _make_worker(follower=follower, leader=leader, stop_event=stop_event)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        leader.connect.assert_called_once()
        leader.disconnect.assert_called_once()

    def test_no_leader_connect_when_leader_is_none(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 1)

        worker = _make_worker(follower=follower, leader=None, stop_event=stop_event)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        follower.disconnect.assert_called_once()

    def test_sets_loaded_event_after_connect(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 1)

        worker = _make_worker(follower=follower, stop_event=stop_event)
        assert worker.loaded_event.is_set()

    def test_stores_follower_state_in_shared_memory(self):
        stop_event = mp.Event()
        follower_state = {"joint1": 1.1, "joint2": 2.2, "joint3": 3.3}
        follower = _make_client()
        call_count = 0

        def read_state():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop_event.set()
            return {"state": follower_state}

        follower.read_state.side_effect = read_state

        worker = _make_worker(follower=follower, stop_event=stop_event)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        assert worker.get_state() == [1.1, 2.2, 3.3]

    def test_from_leader_sends_leader_state_to_follower(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)
        leader_state = {"joint1": 10.0, "joint2": 20.0, "joint3": 30.0}
        leader = _make_client(state=leader_state)

        worker = _make_worker(follower=follower, leader=leader, stop_event=stop_event)
        worker.set_action_read_state(ActionReadState.TELEOPERATION)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        follower.set_joints_state.assert_called()
        assert worker.get_actions() == [10.0, 20.0, 30.0]

    def test_from_leader_ignored_when_no_leader(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)

        worker = _make_worker(follower=follower, leader=None, stop_event=stop_event)
        worker.set_action_read_state(ActionReadState.TELEOPERATION)
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        follower.set_joints_state.assert_not_called()

    def test_from_actions_sends_stored_actions_to_follower(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)

        worker = _make_worker(follower=follower, stop_event=stop_event)
        worker.set_action_read_state(ActionReadState.FROM_ACTIONS)
        worker._set_actions([7.0, 8.0, 9.0])
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        expected = {"joint1": 7.0, "joint2": 8.0, "joint3": 9.0}
        follower.set_joints_state.assert_called()
        assert follower.set_joints_state.call_args[0][0] == expected

    def test_action_source_none_does_not_write_to_follower(self):
        stop_event = mp.Event()
        follower = _make_client()
        follower.read_state.side_effect = _stop_after(stop_event, 2)
        leader = _make_client()

        worker = _make_worker(follower=follower, leader=leader, stop_event=stop_event)
        # action_source stays NONE
        with patch("workers.teleoperate_worker.run_at_frequency", _noop_frequency):
            asyncio.run(worker.run_loop())

        follower.set_joints_state.assert_not_called()
        leader.read_state.assert_not_called()

    def test_wait_for_loading_to_complete_returns_once_event_set(self):
        worker = _make_worker()

        async def _set_and_wait():
            worker.loaded_event.set()
            await worker.wait_for_loading_to_complete()

        asyncio.run(_set_and_wait())

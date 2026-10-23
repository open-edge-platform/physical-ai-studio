import threading

import numpy as np
import pytest
from physicalai.runtime import RobotRuntime

from runtime.action_source import StudioActionSource
from runtime.contract import InMemoryCommandMailbox, QueueEventSink, SetFollowerSourceCommand

from .fakes import FakeObservation, FakeRobot


def test_stop_signal_ends_runtime_with_stop_requested() -> None:
    observation = FakeObservation(np.array([1], dtype=np.float32), timestamp=1)
    follower = FakeRobot([observation])
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=InMemoryCommandMailbox(),
        event_sink=QueueEventSink(),
        fps=30,
    )
    runtime = RobotRuntime(robot=follower, action_source=source, fps=30)
    stop = threading.Event()
    stop.set()

    runtime.connect()
    runtime.run(stop_event=stop)

    assert runtime.last_run_reason == "stop_requested"


def test_follower_connect_error_propagates() -> None:
    follower = FakeRobot([FakeObservation(np.array([1]), timestamp=1)], connect_error="connect failed")
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=InMemoryCommandMailbox(),
        event_sink=QueueEventSink(),
        fps=30,
    )
    runtime = RobotRuntime(robot=follower, action_source=source, fps=30)

    with pytest.raises(ConnectionError, match="connect failed"):
        runtime.connect()


def test_observe_runs_ticks_without_sending_to_the_follower() -> None:
    follower = FakeRobot(positions=[[float(step)] for step in range(1, 20)])
    mailbox = InMemoryCommandMailbox()
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=mailbox,
        event_sink=QueueEventSink(),
        fps=100,
    )
    stop = threading.Event()
    ticks = []

    class _StopAfter:
        def on_tick(self, event) -> None:
            ticks.append(event.action_sent)
            if len(ticks) >= 5:
                stop.set()

    runtime = RobotRuntime(robot=follower, action_source=source, fps=100, callbacks=[_StopAfter()])
    runtime.connect()
    runtime.run(stop_event=stop)

    assert len(ticks) >= 5
    assert all(action is None for action in ticks)
    assert follower.sent_actions == []

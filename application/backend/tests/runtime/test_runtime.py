import threading

import numpy as np
import pytest
from physicalai.runtime import RobotRuntime

from runtime.action_source import StudioActionSource
from runtime.contract import InMemoryCommandMailbox, QueueEventSink

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

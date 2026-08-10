import asyncio
import multiprocessing as mp
import queue
from typing import Any

import pytest
from physicalai.config import Config

from runtime.contract import QueueEventSink
from runtime.hosts.thread_host import RuntimeThreadHost
from runtime.session import RuntimeSession


def _document(*, connect_error: str | None = None) -> dict:
    robot_args: dict[str, Any] = {"positions": [[1.0]], "joint_names": ["joint"]}
    if connect_error is not None:
        robot_args["connect_error"] = connect_error
    return Config(
        "physicalai.runtime.RobotRuntime",
        {
            "robot": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": robot_args,
            },
            "cameras": {},
            "fps": 30.0,
        },
    ).to_dict()


async def test_thread_host_runs_session_until_worker_stop_signal() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    stop = mp.Event()
    stop.set()
    host = RuntimeThreadHost(session, stop_event=stop)

    host.start()
    await host.wait_until_ready()
    await asyncio.to_thread(host.join, 2)

    assert not host.is_alive()
    assert host.error is None
    assert session._runtime is not None
    assert session._runtime.last_run_reason == "stop_requested"
    assert session._follower is not None
    assert not session._follower.is_connected()
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            break
    assert [event.event for event in emitted] == ["state", "lifecycle"]


async def test_thread_host_propagates_setup_error() -> None:
    session = RuntimeSession(_document(connect_error="connect failed"), event_sink=QueueEventSink())
    host = RuntimeThreadHost(session, stop_event=mp.Event())

    host.start()
    with pytest.raises(ConnectionError, match="connect failed"):
        await host.wait_until_ready()
    await asyncio.to_thread(host.join, 2)

    assert not host.is_alive()

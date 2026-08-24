import asyncio
import multiprocessing as mp
import queue
from typing import Any

import pytest
from physicalai.config import Config

from runtime.contract import DisconnectCommand, QueueEventSink
from runtime.hosts.thread_host import RuntimeThreadHost
from runtime.session import RuntimeSession
from tests.runtime.fakes import max_concurrent_connects, reset_connect_tracking


def _fake_robot_args(**kwargs: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"positions": [[1.0]], "joint_names": ["joint"]}
    args.update(kwargs)
    return args


def _document(*, connect_error: str | None = None, **follower_kwargs: Any) -> dict:
    robot_args = _fake_robot_args(**follower_kwargs)
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


def _document_with_leader(*, leader_kwargs: dict[str, Any] | None = None, **follower_kwargs: Any) -> dict:
    document = _document(**follower_kwargs)
    document["init_args"]["action_source"] = {
        "init_args": {
            "leader": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": _fake_robot_args(**(leader_kwargs or {})),
            },
        },
    }
    return document


async def test_thread_host_runs_session_until_worker_stop_signal() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    stop = mp.Event()
    host = RuntimeThreadHost(session, stop_event=stop)

    host.start()
    await host.wait_until_ready()
    stop.set()
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
    assert emitted[0].event == "state"
    assert emitted[-1].event == "lifecycle"


async def test_thread_host_propagates_setup_error() -> None:
    session = RuntimeSession(_document(connect_error="connect failed"), event_sink=QueueEventSink())
    host = RuntimeThreadHost(session, stop_event=mp.Event())

    host.start()
    with pytest.raises(ConnectionError, match="connect failed"):
        await host.wait_until_ready()
    await asyncio.to_thread(host.join, 2)

    assert not host.is_alive()


async def test_preconnect_runs_follower_and_leader_connects_in_parallel() -> None:
    reset_connect_tracking()
    session = RuntimeSession(
        _document_with_leader(
            connect_delay=0.05,
            name="follower",
            leader_kwargs={"connect_delay": 0.05, "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    session._preconnect_robots()

    assert session._follower is not None
    assert session._leader is not None
    assert session._follower.is_connected()
    assert session._leader.is_connected()
    assert max_concurrent_connects() == 2


async def test_preconnect_disconnects_robots_when_one_connect_fails() -> None:
    session = RuntimeSession(
        _document_with_leader(
            name="follower",
            leader_kwargs={"connect_error": "leader connect failed", "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    with pytest.raises(ConnectionError, match="leader connect failed"):
        session._preconnect_robots()

    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()


async def test_disconnect_before_runtime_construction_stops_first_run() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    session.apply(DisconnectCommand())
    await session.setup()

    session.run(mp.Event())

    assert session._runtime is None
    assert session._follower is not None
    assert not session._follower.is_connected()


async def test_worker_stop_before_run_does_not_connect_hardware() -> None:
    session = RuntimeSession(_document_with_leader(), event_sink=QueueEventSink())
    await session.setup()
    stop = mp.Event()
    stop.set()

    session.run(stop)

    assert session._runtime is None
    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()

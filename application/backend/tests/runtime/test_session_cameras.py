from __future__ import annotations

import time
from typing import Any

import pytest

from runtime.contract import QueueEventSink
from runtime.session import RuntimeSession
from tests.runtime.fakes import (
    max_concurrent_connects,
    recorded_disconnects,
    reset_connect_tracking,
    reset_disconnect_tracking,
)
from tests.runtime.test_session import _camera_config, _document_with_cameras, _document_with_leader


def _document_with_leader_and_cameras(
    *names: str,
    leader_kwargs: dict[str, Any] | None = None,
    camera_kwargs: dict[str, dict[str, Any]] | None = None,
    **follower_kwargs: Any,
) -> dict[str, Any]:
    document = _document_with_leader(leader_kwargs=leader_kwargs, **follower_kwargs)
    per_camera = camera_kwargs or {}
    cameras: dict[str, dict[str, Any]] = {}
    for key in names:
        init_args = dict(per_camera.get(key, {}))
        init_args["name"] = key
        cameras[key] = _camera_config(**init_args)
    document["init_args"]["cameras"] = cameras
    return document


async def test_session_instantiates_cameras_from_the_document() -> None:
    document = _document_with_cameras("front", "wrist")
    session = RuntimeSession(document, event_sink=QueueEventSink())
    await session.setup()
    runtime = session.build_runtime()

    assert list(runtime.cameras) == ["front", "wrist"]
    assert session._cameras["front"] is runtime.cameras["front"]
    assert session._cameras["wrist"] is runtime.cameras["wrist"]


async def test_devices_connect_in_parallel() -> None:
    reset_connect_tracking()
    delay = 0.05
    session = RuntimeSession(
        _document_with_leader_and_cameras(
            "front",
            connect_delay=delay,
            name="follower",
            leader_kwargs={"connect_delay": delay, "name": "leader"},
            camera_kwargs={"front": {"connect_delay": delay, "name": "front"}},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    started = time.perf_counter()
    session._preconnect_devices()
    elapsed = time.perf_counter() - started

    assert session._follower is not None
    assert session._leader is not None
    assert session._follower.is_connected()
    assert session._leader.is_connected()
    assert session._cameras["front"].is_connected
    assert max_concurrent_connects() == 3
    assert elapsed < delay * 2.5


async def test_a_failing_camera_rolls_back_every_connected_device() -> None:
    session = RuntimeSession(
        _document_with_leader_and_cameras(
            "front",
            name="follower",
            leader_kwargs={"name": "leader"},
            camera_kwargs={"front": {"connect_error": "camera connect failed", "name": "front"}},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    with pytest.raises(ConnectionError, match="camera connect failed"):
        session._preconnect_devices()

    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()
    assert not session._cameras["front"].is_connected


async def test_teardown_disconnects_cameras_then_robots() -> None:
    reset_disconnect_tracking()
    session = RuntimeSession(
        _document_with_leader_and_cameras(
            "front",
            "wrist",
            name="follower",
            leader_kwargs={"name": "leader"},
            camera_kwargs={
                "front": {"name": "front", "disconnect_error": "front disconnect failed"},
                "wrist": {"name": "wrist"},
            },
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()
    session._preconnect_devices()

    await session.teardown()

    assert recorded_disconnects() == ["front", "wrist", "leader", "follower"]
    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()
    assert not session._cameras["wrist"].is_connected

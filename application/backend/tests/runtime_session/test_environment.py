from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime_session.bus import publish_status
from runtime_session.environment import _enter_concurrently, describe_environment
from runtime_session.store import ImageSpec
from tests.workers.fake_robot import FakeRobot


@pytest.fixture
def factory(mock_robot_client_factory, tmp_path):
    driver = FakeRobot(str(tmp_path / "log"), ["shoulder_pan", "gripper"], [0.0, 0.0])
    mock_robot_client_factory.build_robot_driver = AsyncMock(return_value=(driver, object()))
    return mock_robot_client_factory


async def test_a_slow_device_cannot_overwrite_a_connected_cameras_reconnect_warning() -> None:
    bus = MagicMock()
    camera_connected = asyncio.Event()
    camera_exited = MagicMock()

    @asynccontextmanager
    async def camera():
        camera_connected.set()
        try:
            yield "camera"
        finally:
            camera_exited()

    @asynccontextmanager
    async def slow_device():
        await camera_connected.wait()
        publish_status(bus, "environment/cam", "starting", "no frames; reconnecting")
        yield "follower"

    async with AsyncExitStack() as stack:
        started = await _enter_concurrently(stack, bus, {"cam": camera(), "follower": slow_device()})
        assert started == {"cam": "camera", "follower": "follower"}
        statuses = [call.args[1] for call in bus.publish.call_args_list if call.args[1].component == "environment/cam"]
        assert [status.state for status in statuses] == ["starting", "running", "starting"]
        assert statuses[-1].error == "no frames; reconnecting"
        camera_exited.assert_not_called()
    camera_exited.assert_called_once()


async def test_spec_comes_from_the_driver_and_the_camera_rows(test_environment, factory) -> None:
    description = await describe_environment(test_environment, factory)

    assert description.spec.joint_names == ("shoulder_pan.pos", "gripper.pos")
    assert description.spec.observations == {"grabber": ImageSpec(480, 640), "front": ImageSpec(480, 640)}
    assert description.follower.robot_id == test_environment.robots[0].robot.id
    assert set(description.cameras) == {"grabber", "front"}


async def test_more_than_one_robot_is_refused(test_environment, factory) -> None:
    test_environment.robots.append(test_environment.robots[0])
    with pytest.raises(ValueError, match="has 2 robots"):
        await describe_environment(test_environment, factory)


async def test_cameras_that_would_share_a_key_are_refused(test_environment, factory) -> None:
    test_environment.cameras[1].name = "Grabber"
    with pytest.raises(ValueError, match="'grabber'"):
        await describe_environment(test_environment, factory)

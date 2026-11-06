from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from runtime_session.environment import describe_environment
from runtime_session.store import ImageSpec
from tests.workers.fake_robot import FakeRobot


@pytest.fixture
def factory(mock_robot_client_factory, tmp_path):
    driver = FakeRobot(str(tmp_path / "log"), ["shoulder_pan", "gripper"], [0.0, 0.0])
    mock_robot_client_factory.build_robot_driver = AsyncMock(return_value=(driver, object()))
    return mock_robot_client_factory


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


async def test_a_leader_can_refuse_the_follower(test_environment, factory, tmp_path) -> None:
    follower = FakeRobot(str(tmp_path / "log"), ["shoulder_pan", "gripper"], [0.0, 0.0])
    seen = []

    class Leader:  # answers in the twin's units, as a headset leader does
        def check_follower(self, driver: object) -> None:
            seen.append(driver)
            raise ValueError("the follower is set to degrees; set the follower to normalized")

    factory.build_robot_driver = AsyncMock(side_effect=[(follower, object()), (Leader(), object())])
    configuration = test_environment.robots[0]
    object.__setattr__(configuration, "tele_operator", SimpleNamespace(robot=configuration.robot))

    with pytest.raises(ValueError, match="set the follower to normalized"):
        await describe_environment(test_environment, factory)
    assert seen == [follower]

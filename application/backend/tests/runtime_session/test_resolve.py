from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from runtime_session.resolve import project_resolvers
from schemas.project_camera import IPCamera, SimulationCamera
from schemas.runtime_session import EnvironmentDraft
from tests.runtime_session.fixtures import FOLLOWER_ID, LEADER_ID, make_environment

PROJECT_ID = uuid4()
ENVIRONMENT = make_environment(cameras=0)
FOLLOWER = ENVIRONMENT.robots[0].robot
LEADER = ENVIRONMENT.robots[0].tele_operator.robot


def _resolvers(
    simulations: object | None = None, cameras: object | None = None, robot_simulations: object | None = None
):
    rows = {FOLLOWER_ID: FOLLOWER, LEADER_ID: LEADER}
    robots = SimpleNamespace(get_robot_by_id=AsyncMock(side_effect=lambda _project, robot_id: rows[robot_id]))
    return project_resolvers(
        PROJECT_ID,
        robots,
        cameras or SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        simulations,
        robot_simulations,
    )


def _draft(*, leader: bool = True, camera_ids: list[UUID] | None = None) -> EnvironmentDraft:
    tele_operator = {"type": "robot", "robot_id": str(LEADER_ID)} if leader else {"type": "none"}
    return EnvironmentDraft.model_validate(
        {
            "name": "Bench",
            "robots": [{"robot_id": str(FOLLOWER_ID), "tele_operator": tele_operator}],
            "cameras": [{"camera_id": str(camera_id)} for camera_id in camera_ids or []],
        }
    )


async def test_the_leader_is_pointed_at_the_followers_simulation() -> None:
    pointed = LEADER.model_copy(update={"name": "Leader on the simulation's port"})
    simulations = SimpleNamespace(point_leader_at_follower=AsyncMock(return_value=(FOLLOWER, pointed)))

    environment = await _resolvers(simulations).environment(_draft())

    simulations.point_leader_at_follower.assert_awaited_once_with(PROJECT_ID, FOLLOWER, LEADER)
    assert environment.robots[0].tele_operator.robot == pointed


async def test_an_environment_without_a_leader_is_not_pointed() -> None:
    simulations = SimpleNamespace(point_leader_at_follower=AsyncMock())

    environment = await _resolvers(simulations).environment(_draft(leader=False))

    simulations.point_leader_at_follower.assert_not_awaited()
    assert environment.robots[0].robot == FOLLOWER


async def test_simulation_cameras_stream_from_where_their_simulation_runs_now() -> None:
    stored = SimulationCamera(
        id=uuid4(),
        name="Wrist",
        fingerprint={"robot_id": str(FOLLOWER_ID), "camera": "wrist"},
        hardware_name="wrist",
        payload={"robot_id": FOLLOWER_ID, "camera": "wrist"},
    )
    streaming = IPCamera(
        id=stored.id,
        name=stored.name,
        fingerprint=stored.fingerprint,
        hardware_name="wrist",
        payload={"url": "http://127.0.0.1:50001/cameras/wrist/mjpeg"},
    )
    cameras = SimpleNamespace(get_camera_by_id=AsyncMock(return_value=stored))
    robot_simulations = SimpleNamespace(resolve_cameras=AsyncMock(return_value=[streaming]))

    environment = await _resolvers(cameras=cameras, robot_simulations=robot_simulations).environment(
        _draft(leader=False, camera_ids=[stored.id])
    )

    robot_simulations.resolve_cameras.assert_awaited_once_with(PROJECT_ID, [stored])
    assert environment.cameras == [streaming]

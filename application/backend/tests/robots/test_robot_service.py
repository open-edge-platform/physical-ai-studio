from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from db.schema import Base, ProjectDB
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.robot import RobotAdapter

PROJECT_ID = uuid4()
SO101 = "MuJoCo_SO101_Follower"
SO101_DEFAULTS = {"name": "mujoco-so101-follow", "http_url": "http://127.0.0.1:8080"}
"""What the robot form sends for it: every field's schema default."""


@pytest.fixture
async def service() -> AsyncIterator[RobotService]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            session.add(ProjectDB(id=str(PROJECT_ID), name="Project"))
            await session.commit()
            yield RobotService(session, RobotCatalogRegistry())
    finally:
        await engine.dispose()


def _robot(robot_type: str, payload: dict[str, Any], robot_id: UUID | None = None) -> Any:
    return RobotAdapter.validate_python(
        {"id": str(robot_id or uuid4()), "name": "Arm", "type": robot_type, "payload": payload}
    )


async def _create(service: RobotService, robot_type: str, payload: dict[str, Any], robot_id: UUID | None = None) -> Any:
    robot = _robot(robot_type, payload, robot_id)
    await service.create_robot(PROJECT_ID, robot)
    return await service.get_robot_by_id(PROJECT_ID, robot.id)


def _port(robot: Any) -> int | None:
    return urlsplit(robot.payload.http_url).port


@pytest.mark.parametrize("payload", [SO101_DEFAULTS, {}], ids=["form defaults", "omitted"])
async def test_a_new_simulated_robot_on_the_defaults_gets_a_simulation_of_its_own(
    service: RobotService, payload: dict[str, Any]
) -> None:
    robot = await _create(service, SO101, payload)

    assert robot.payload.name == f"mujoco-so101-follow-{UUID(str(robot.id)).hex}"
    url = urlsplit(robot.payload.http_url)
    assert (url.scheme, url.hostname) == ("http", "127.0.0.1")
    assert url.port not in {None, 8080}


async def test_ids_made_close_in_time_still_get_their_own_names(service: RobotService) -> None:
    # Time-ordered ids (UUIDv7) of robots made together share their first digits.
    first = await _create(service, SO101, SO101_DEFAULTS, UUID("01a1cf7d-f531-7000-8000-000000000001"))
    second = await _create(service, SO101, SO101_DEFAULTS, UUID("01a1cf7d-f531-7000-8000-000000000002"))

    assert first.payload.name != second.payload.name


async def test_a_bimanual_robot_gets_its_own_name_from_its_own_default(service: RobotService) -> None:
    robot = await _create(service, "MuJoCo_SO101_Bimanual_Follower", {})

    assert robot.payload.name == f"mujoco-so101-bimanual-follow-{UUID(str(robot.id)).hex}"


@pytest.mark.parametrize(
    ("payload", "name", "port"),
    [
        ({"name": "my-sim", "http_url": "http://127.0.0.1:8173"}, "my-sim", 8173),
        ({"name": "my-sim", "http_url": "http://127.0.0.1:8080"}, "my-sim", None),
        ({"name": "mujoco-so101-follow", "http_url": "http://127.0.0.1:8173"}, None, 8173),
    ],
    ids=["both set", "name set", "address set"],
)
async def test_each_value_the_user_set_stays(
    service: RobotService, payload: dict[str, Any], name: str | None, port: int | None
) -> None:
    robot = await _create(service, SO101, payload)

    # ``None``: left on its default, so made unique.
    assert robot.payload.name == (name or f"mujoco-so101-follow-{UUID(str(robot.id)).hex}")
    if port is not None:
        assert _port(robot) == port
    else:
        assert _port(robot) not in {None, 8080}


async def test_an_update_keeps_the_defaults(service: RobotService) -> None:
    robot = await _create(service, SO101, {"name": "my-sim", "http_url": "http://127.0.0.1:8173"})

    # Pointing an existing robot at a simulation on the defaults is how to attach to one.
    await service.update_robot(
        PROJECT_ID, robot.model_copy(update={"payload": robot.payload.model_validate(SO101_DEFAULTS)})
    )
    updated = await service.get_robot_by_id(PROJECT_ID, robot.id)

    assert (updated.payload.name, updated.payload.http_url) == (SO101_DEFAULTS["name"], SO101_DEFAULTS["http_url"])


async def test_a_robot_without_a_simulation_keeps_its_payload(service: RobotService) -> None:
    robot = await _create(service, "MuJoCo_Virtual_Leader", {})

    assert robot.payload.http_port == 8080

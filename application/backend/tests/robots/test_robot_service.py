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


async def _create(service: RobotService, robot_type: str, payload: dict[str, Any]) -> Any:
    robot_id = uuid4()
    await service.create_robot(
        PROJECT_ID,
        RobotAdapter.validate_python({"id": str(robot_id), "name": "Arm", "type": robot_type, "payload": payload}),
    )
    return await service.get_robot_by_id(PROJECT_ID, robot_id)


async def test_a_new_simulated_robot_on_the_defaults_gets_a_simulation_of_its_own(service: RobotService) -> None:
    first = await _create(service, "MuJoCo_SO101_Follower", {})
    second = await _create(service, "MuJoCo_SO101_Follower", {})

    # Each attaches to its own simulation, not to another robot's (or a terminal's) under the defaults.
    assert first.payload.name == f"mujoco-so101-follow-{UUID(str(first.id)).hex[:8]}"
    assert second.payload.name != first.payload.name
    first_url, second_url = urlsplit(first.payload.http_url), urlsplit(second.payload.http_url)
    assert (first_url.scheme, first_url.hostname) == ("http", "127.0.0.1")
    assert first_url.port not in {None, 8080}
    assert second_url.port not in {None, 8080}


async def test_a_name_and_address_the_user_set_stay(service: RobotService) -> None:
    robot = await _create(service, "MuJoCo_SO101_Follower", {"name": "my-sim", "http_url": "http://127.0.0.1:8173"})

    assert robot.payload.name == "my-sim"
    assert robot.payload.http_url == "http://127.0.0.1:8173"


async def test_a_robot_without_a_simulation_keeps_its_payload(service: RobotService) -> None:
    robot = await _create(service, "MuJoCo_Virtual_Leader", {})

    assert robot.payload.http_port == 8080

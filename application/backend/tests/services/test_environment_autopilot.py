"""The simulation's autopilot as an environment's teleoperator: only for a simulated follower, stored as a choice."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.schema import Base, EnvironmentRobotDB, ProjectDB
from exceptions import SimulationNotSupportedError
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.environment import Environment, TeleoperatorAutopilot, TeleoperatorAutopilotWithRobot
from schemas.robot import RobotAdapter
from services.environment_service import EnvironmentService

PROJECT_ID = uuid4()
SIMULATED_ID = uuid4()
HARDWARE_ID = uuid4()
REGISTRY = RobotCatalogRegistry()


@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        session.add(ProjectDB(id=str(PROJECT_ID), name="Project"))
        await session.commit()
        robots = RobotService(session, REGISTRY)
        for robot_id, robot_type, payload in (
            (SIMULATED_ID, "MuJoCo_SO101_Follower", {"http_url": "http://127.0.0.1:50001"}),
            (HARDWARE_ID, "SO101_Follower", {"connection_string": "", "serial_number": "A"}),
        ):
            await robots.create_robot(
                PROJECT_ID,
                RobotAdapter.validate_python(
                    {"id": str(robot_id), "name": robot_type, "type": robot_type, "payload": payload}
                ),
            )
        yield session
    await engine.dispose()


def _environment(robot_id: UUID, tele_operator: dict[str, Any]) -> Environment:
    return Environment.model_validate(
        {"id": str(uuid4()), "name": "Rig", "robots": [{"robot_id": str(robot_id), "tele_operator": tele_operator}]}
    )


async def test_a_simulated_follower_is_led_by_its_autopilot_and_nothing_else_is_stored(db: AsyncSession) -> None:
    service = EnvironmentService(db, REGISTRY)
    environment = _environment(SIMULATED_ID, {"type": "autopilot"})

    await service.create_environment(PROJECT_ID, environment)

    link = (await db.scalars(select(EnvironmentRobotDB))).one()
    assert (link.tele_operator_type, link.tele_operator_robot_id) == ("autopilot", None)
    assert (await service.get_environment_list(PROJECT_ID))[0].robots[0].tele_operator == TeleoperatorAutopilot()
    loaded = await service.get_environment_by_id(PROJECT_ID, environment.id)
    assert loaded.robots[0].tele_operator == TeleoperatorAutopilotWithRobot(robot=None)


async def test_a_follower_without_a_simulation_cannot_be_led_by_an_autopilot(db: AsyncSession) -> None:
    service = EnvironmentService(db, REGISTRY)

    with pytest.raises(SimulationNotSupportedError, match="has no simulation") as error:
        await service.create_environment(PROJECT_ID, _environment(HARDWARE_ID, {"type": "autopilot"}))

    assert error.value.http_status == 400
    assert (await db.scalars(select(EnvironmentRobotDB))).all() == []


async def test_an_update_cannot_give_a_follower_without_a_simulation_an_autopilot(db: AsyncSession) -> None:
    service = EnvironmentService(db, REGISTRY)
    environment = _environment(HARDWARE_ID, {"type": "none"})
    await service.create_environment(PROJECT_ID, environment)

    with pytest.raises(SimulationNotSupportedError):
        await service.update_environment(
            PROJECT_ID, _environment(HARDWARE_ID, {"type": "autopilot"}).model_copy(update={"id": environment.id})
        )

    assert (await db.scalars(select(EnvironmentRobotDB.tele_operator_type))).all() == ["none"]

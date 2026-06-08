"""Round-trip tests for the environment repository backed by the new join tables.

These exercise the real mapper + repository against an in-memory SQLite database to verify that:
* robots/cameras are persisted into ``environment_robots`` / ``environment_cameras`` join rows,
* ``get_by_id_with_relations`` eager-loads them and overrides each robot/camera name with the
  per-environment name,
* ``update`` fully replaces the join rows (rename + add/remove) without leaving orphans.
"""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from db.schema import Base, EnvironmentCameraDB, EnvironmentRobotDB, ProjectCameraDB, ProjectDB, ProjectRobotDB
from repositories.project_environment_repo import ProjectEnvironmentRepository
from schemas.environment import (
    CameraEnvironmentConfiguration,
    Environment,
    RobotEnvironmentConfiguration,
    TeleoperatorNone,
    TeleoperatorRobot,
)
from schemas.robot import RobotType

PROJECT_ID = str(uuid4())
FOLLOWER_ID = str(uuid4())
LEADER_ID = str(uuid4())
CAM_FRONT_ID = str(uuid4())
CAM_WRIST_ID = str(uuid4())


def _seed_objects() -> list:
    return [
        ProjectDB(id=PROJECT_ID, name="Project"),
        ProjectRobotDB(
            id=FOLLOWER_ID,
            project_id=PROJECT_ID,
            name="Khaos",
            type=RobotType.SO101_FOLLOWER,
            payload={"serial_number": "F1"},
        ),
        ProjectRobotDB(
            id=LEADER_ID,
            project_id=PROJECT_ID,
            name="Nyx",
            type=RobotType.SO101_LEADER,
            payload={"serial_number": "L1"},
        ),
        ProjectCameraDB(
            id=CAM_FRONT_ID,
            project_id=PROJECT_ID,
            name="grabber",
            driver="usb_camera",
            fingerprint="/dev/video0",
            hardware_name=None,
            payload={"width": 640, "height": 480, "fps": 30},
        ),
        ProjectCameraDB(
            id=CAM_WRIST_ID,
            project_id=PROJECT_ID,
            name="webcam",
            driver="usb_camera",
            fingerprint="/dev/video1",
            hardware_name=None,
            payload={"width": 640, "height": 480, "fps": 30},
        ),
    ]


async def _setup_session() -> tuple[AsyncSession, object]:
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add_all(_seed_objects())
        await session.commit()
    return session_factory, engine


def _new_environment() -> Environment:
    return Environment.model_validate(
        {
            "id": str(uuid4()),
            "name": "Home Setup",
            "robots": [
                {
                    "robot_id": FOLLOWER_ID,
                    "name": "primary",
                    "tele_operator": {"type": "robot", "robot_id": LEADER_ID},
                },
            ],
            "cameras": [
                {"camera_id": CAM_FRONT_ID, "name": "front"},
                {"camera_id": CAM_WRIST_ID, "name": "wrist"},
            ],
        }
    )


def test_save_and_get_with_relations_overrides_names() -> None:
    async def _run() -> None:
        session_factory, engine = await _setup_session()
        env = _new_environment()
        try:
            async with session_factory() as session:
                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                await repo.save(env)

            async with session_factory() as session:
                # Join rows landed in the dedicated tables.
                robot_links = (await session.execute(select(EnvironmentRobotDB))).scalars().all()
                camera_links = (await session.execute(select(EnvironmentCameraDB))).scalars().all()
                assert len(robot_links) == 1
                assert len(camera_links) == 2

                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                loaded = await repo.get_by_id_with_relations(env.id)

            assert loaded is not None
            # Per-environment names override the underlying robot/camera names.
            assert loaded.robots[0].robot.name == "primary"
            assert {c.name for c in loaded.cameras} == {"front", "wrist"}
            # Teleoperator robot is eager-loaded and keeps its own name.
            assert loaded.robots[0].tele_operator.type == "robot"
            assert loaded.robots[0].tele_operator.robot.name == "Nyx"
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_update_replaces_join_rows() -> None:
    async def _run() -> None:
        session_factory, engine = await _setup_session()
        env = _new_environment()
        try:
            async with session_factory() as session:
                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                await repo.save(env)

            # Rename the robot, drop the wrist camera, rename the front camera, drop the teleoperator.
            updated = env.model_copy(
                update={
                    "robots": [
                        RobotEnvironmentConfiguration(
                            robot_id=FOLLOWER_ID, name="renamed", tele_operator=TeleoperatorNone()
                        )
                    ],
                    "cameras": [CameraEnvironmentConfiguration(camera_id=CAM_FRONT_ID, name="main")],
                }
            )

            async with session_factory() as session:
                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                existing = await repo.get_by_id(env.id)
                await repo.update(existing, updated.model_dump(exclude={"id", "created_at", "updated_at"}))

            async with session_factory() as session:
                # No orphan join rows remain.
                robot_count = (await session.execute(select(func.count()).select_from(EnvironmentRobotDB))).scalar()
                camera_count = (await session.execute(select(func.count()).select_from(EnvironmentCameraDB))).scalar()
                assert robot_count == 1
                assert camera_count == 1

                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                loaded = await repo.get_by_id_with_relations(env.id)

            assert loaded is not None
            assert loaded.robots[0].robot.name == "renamed"
            assert loaded.robots[0].tele_operator.type == "none"
            assert [c.name for c in loaded.cameras] == ["main"]
        finally:
            await engine.dispose()

    asyncio.run(_run())


@pytest.mark.parametrize("teleop", [TeleoperatorNone(), TeleoperatorRobot(robot_id=LEADER_ID)])
def test_lightweight_round_trip_preserves_names(teleop: object) -> None:
    async def _run() -> None:
        session_factory, engine = await _setup_session()
        env = Environment.model_validate(
            {
                "id": str(uuid4()),
                "name": "Env",
                "robots": [{"robot_id": FOLLOWER_ID, "name": "primary", "tele_operator": teleop.model_dump()}],
                "cameras": [{"camera_id": CAM_FRONT_ID, "name": "front"}],
            }
        )
        try:
            async with session_factory() as session:
                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                await repo.save(env)

            async with session_factory() as session:
                repo = ProjectEnvironmentRepository(session, PROJECT_ID)
                loaded = await repo.get_by_id(env.id)

            assert loaded is not None
            assert loaded.robots[0].name == "primary"
            assert loaded.robots[0].tele_operator == teleop
            assert loaded.cameras[0].name == "front"
            assert loaded.cameras[0].camera_id == env.cameras[0].camera_id
        finally:
            await engine.dispose()

    asyncio.run(_run())

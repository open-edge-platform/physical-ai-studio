# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Robot payloads must be JSON-compatible at the database boundary."""

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from db.schema import Base, ProjectDB, ProjectRobotDB
from repositories.mappers import project_robot_mapper
from repositories.mappers.project_robot_mapper import ProjectRobotMapper
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.robot_type import BaseRobot


class PathPayload(BaseModel):
    calibration_dir: Path | None = None
    camera_paths: dict[str, list[Path]] = Field(default_factory=dict)


class PathRobot(BaseRobot):
    type: Literal["path_test_follower"] = "path_test_follower"
    payload: PathPayload


@pytest.mark.parametrize("calibration_dir", [None, Path("/calibration/stararm")])
def test_to_schema_serializes_paths_without_mutating_payload(calibration_dir):
    robot = PathRobot(
        id=uuid4(),
        name="Path test",
        payload=PathPayload(calibration_dir=calibration_dir, camera_paths={"front": [Path("/dev/video0")]}),
    )

    model = ProjectRobotMapper.to_schema(robot)

    assert json.loads(json.dumps(model.payload)) == {
        "calibration_dir": None if calibration_dir is None else str(calibration_dir),
        "camera_paths": {"front": ["/dev/video0"]},
    }
    assert robot.payload.calibration_dir == calibration_dir
    assert isinstance(robot.payload.camera_paths["front"][0], Path)


@pytest.mark.asyncio
async def test_create_and_update_persist_path_payloads(monkeypatch):
    # Use a local test schema so the regression does not depend on optional
    # LeRobot extensions or trust settings, and never opens hardware ports.
    monkeypatch.setattr(project_robot_mapper, "RobotAdapter", TypeAdapter(PathRobot))
    registry = MagicMock(spec=RobotCatalogRegistry)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with AsyncSession(engine, expire_on_commit=False) as session:
            project_id = uuid4()
            session.add(ProjectDB(id=str(project_id), name="Path test project"))
            await session.commit()
            service = RobotService(session, registry)
            robot = PathRobot(
                id=uuid4(),
                name="Path test",
                payload=PathPayload(calibration_dir=Path("/calibration/original")),
            )

            await service.create_robot(project_id, robot)
            stored = await session.get(ProjectRobotDB, str(robot.id))
            assert stored.payload["calibration_dir"] == "/calibration/original"

            updated = robot.model_copy(update={"payload": PathPayload(calibration_dir=Path("/calibration/updated"))})
            restored = await service.update_robot(project_id, updated)
            await session.refresh(stored)

            assert stored.payload["calibration_dir"] == "/calibration/updated"
            assert restored.payload.calibration_dir == Path("/calibration/updated")
            assert isinstance(restored.payload.calibration_dir, Path)
    finally:
        await engine.dispose()

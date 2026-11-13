# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deleting a robot whose simulation cameras a runtime session uses."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from api.robots import delete_project_robot
from exceptions import RecordingLockError
from schemas.project_camera import SimulationCamera

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()


def _camera(robot_id) -> SimulationCamera:
    return SimulationCamera(
        id=uuid4(),
        name="Sim arm wrist",
        fingerprint={"robot_id": str(robot_id), "camera": "wrist"},
        hardware_name="wrist",
        payload={"robot_id": robot_id, "camera": "wrist"},
    )


async def _delete(cameras: list[SimulationCamera], claimed: SimulationCamera | None) -> AsyncMock:
    robot_service = SimpleNamespace(
        get_robot_by_id=AsyncMock(return_value=SimpleNamespace(id=ROBOT_ID, name="Sim arm")),
        delete_robot=AsyncMock(),
    )
    claims = SimpleNamespace(
        holder_of=lambda fingerprint: (
            SimpleNamespace(project_name="Other")
            if claimed is not None and fingerprint == claimed.fingerprint
            else None
        )
    )
    await delete_project_robot(
        project_id=PROJECT_ID,
        robot_id=ROBOT_ID,
        robot_service=robot_service,  # type: ignore[arg-type]
        camera_service=SimpleNamespace(get_camera_list=AsyncMock(return_value=cameras)),  # type: ignore[arg-type]
        registry=MagicMock(),
        claims=claims,  # type: ignore[arg-type]
    )
    return robot_service.delete_robot


async def test_a_robot_whose_simulation_camera_is_in_use_is_not_deleted() -> None:
    mine = _camera(ROBOT_ID)

    with pytest.raises(RecordingLockError, match="Sim arm wrist"):
        await _delete([mine], claimed=mine)


async def test_another_robots_camera_in_use_does_not_stop_the_deletion() -> None:
    theirs = _camera(uuid4())

    delete = await _delete([_camera(ROBOT_ID), theirs], claimed=theirs)

    delete.assert_awaited_once_with(PROJECT_ID, ROBOT_ID)

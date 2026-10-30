# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Routes of a robot's simulation against a stub service: status codes and error codes."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.robot_simulation import get_robot_simulation_service
from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationCommandUnavailableError,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotSupportedError,
)
from main import app
from schemas.simulation import RobotSimulation, SimulationCommand

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
ENVIRONMENT_ID = uuid4()


class StubService:
    error: Exception | None = None

    async def get(self, project_id, robot_id) -> RobotSimulation:
        if self.error is not None:
            raise self.error
        return RobotSimulation(running=True, started_by="external", owner_name="sim", scene="yahtzee")

    async def command(self, project_id, robot_id, scene, seed) -> SimulationCommand:
        self.commanded = (scene, seed)
        if self.error is not None:
            raise self.error
        return SimulationCommand(argv=["sim", "start", "--scene", scene], command=f"sim start --scene {scene}")

    async def add_cameras(self, project_id, environment_id, robot_id, names, camera_names) -> list:
        self.added = (names, camera_names)
        if self.error is not None:
            raise self.error
        return []


@pytest.fixture
def stub() -> Iterator[StubService]:
    stub = StubService()
    app.dependency_overrides[get_robot_simulation_service] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


def test_get_returns_the_robot_simulation(stub: StubService) -> None:
    response = TestClient(app).get(f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation")

    assert response.status_code == 200
    assert response.json()["started_by"] == "external"


def test_add_cameras_returns_201(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"], "names": {"wrist": "Wrist"}},
    )

    assert response.status_code == 201
    assert stub.added == (["wrist"], {"wrist": "Wrist"})


def test_add_cameras_names_are_optional(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"]},
    )

    assert response.status_code == 201
    assert stub.added == (["wrist"], {})


def test_add_cameras_needs_at_least_one_camera(stub: StubService) -> None:
    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": []},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (SimulationNotSupportedError("no simulation"), 400, "simulation_not_supported"),
        (ResourceNotFoundError(ResourceType.ROBOT, ROBOT_ID), 404, "Robot_not_found"),
        (SimulationNotRunningError("not running"), 503, "simulation_not_running"),
        (SimulationInvalidRequestError("no camera named x"), 400, "simulation_invalid_request"),
    ],
)
def test_errors_map_to_their_codes(stub: StubService, error: Exception, status_code: int, error_code: str) -> None:
    stub.error = error

    response = TestClient(app).post(
        f"/api/projects/{PROJECT_ID}/environments/{ENVIRONMENT_ID}/simulation-cameras",
        json={"robot_id": str(ROBOT_ID), "cameras": ["wrist"]},
    )

    assert response.status_code == status_code
    assert response.json()["error_code"] == error_code


def test_command_returns_the_command_for_the_scene_and_seed(stub: StubService) -> None:
    response = TestClient(app).get(
        f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/command", params={"scene": "yahtzee", "seed": 7}
    )

    assert response.status_code == 200
    assert response.json() == {"argv": ["sim", "start", "--scene", "yahtzee"], "command": "sim start --scene yahtzee"}
    assert stub.commanded == ("yahtzee", 7)


def test_command_seed_is_optional(stub: StubService) -> None:
    response = TestClient(app).get(
        f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/command", params={"scene": "yahtzee"}
    )

    assert response.status_code == 200
    assert stub.commanded == ("yahtzee", None)


@pytest.mark.parametrize("params", [{}, {"scene": "yahtzee", "seed": -1}])
def test_command_needs_a_scene_and_a_non_negative_seed(stub: StubService, params: dict) -> None:
    response = TestClient(app).get(f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/command", params=params)

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (SimulationCommandUnavailableError("no command"), 404, "simulation_command_unavailable"),
        (SimulationInvalidRequestError("Unknown scene"), 400, "simulation_invalid_request"),
        (SimulationNotSupportedError("no simulation"), 400, "simulation_not_supported"),
    ],
)
def test_command_errors_map_to_their_codes(
    stub: StubService, error: Exception, status_code: int, error_code: str
) -> None:
    stub.error = error

    response = TestClient(app).get(
        f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation/command", params={"scene": "yahtzee"}
    )

    assert response.status_code == status_code
    assert response.json()["error_code"] == error_code

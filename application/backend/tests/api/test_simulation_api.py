# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Routes of /api/simulation against a stub manager: payloads, status codes and error codes."""

import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api.robot_catalog as robot_catalog
from api.dependencies import get_simulation_manager
from api.simulation import simulation_events
from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationBusyError,
    SimulationInvalidRequestError,
    SimulationNameInUseError,
    SimulationNotSupportedError,
)
from main import app
from schemas.simulation import Simulation, SimulationStatus, StartSimulationRequest

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
STARTING = Simulation(
    status=SimulationStatus.STARTING, project_id=PROJECT_ID, robot_id=ROBOT_ID, owner_name="sim-a", scene="tabletop"
)


class StubManager:
    def __init__(self) -> None:
        self.current = Simulation()
        self.start_error: Exception | None = None
        self.started: list[StartSimulationRequest] = []
        self.stopped = 0

    def state(self) -> Simulation:
        return self.current

    def logs(self) -> list[str]:
        return ["loading scene", "ready"]

    async def start(self, request: StartSimulationRequest) -> Simulation:
        if self.start_error is not None:
            raise self.start_error
        self.started.append(request)
        self.current = STARTING
        return self.current

    async def stop(self) -> Simulation:
        self.stopped += 1
        self.current = Simulation()
        return self.current

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator:
        import asyncio

        queue: asyncio.Queue[Simulation] = asyncio.Queue()
        queue.put_nowait(Simulation())
        queue.put_nowait(STARTING)
        yield queue


@pytest.fixture
def manager() -> Iterator[StubManager]:
    stub = StubManager()
    app.dependency_overrides[get_simulation_manager] = lambda: stub
    yield stub
    app.dependency_overrides.clear()


@pytest.fixture
def client(manager: StubManager) -> TestClient:
    return TestClient(app)


def start_body(**changes: object) -> dict[str, object]:
    return {"project_id": str(PROJECT_ID), "robot_id": str(ROBOT_ID), "scene": "tabletop", **changes}


def test_get_returns_idle_when_nothing_runs(client: TestClient) -> None:
    response = client.get("/api/simulation")

    assert response.status_code == 200
    assert response.json() == {
        "status": "idle",
        "project_id": None,
        "robot_id": None,
        "robot_type": None,
        "owner_name": None,
        "scene": None,
        "seed": None,
        "phase": None,
        "download": None,
        "pid": None,
        "started_at": None,
        "http_url": None,
        "viewer_url": None,
        "cameras": [],
        "error": None,
    }


def test_post_starts_and_returns_202_starting(client: TestClient, manager: StubManager) -> None:
    response = client.post("/api/simulation", json=start_body(seed=11, replace=True))

    assert response.status_code == 202
    assert response.json()["status"] == "starting"
    assert manager.started == [
        StartSimulationRequest(project_id=PROJECT_ID, robot_id=ROBOT_ID, scene="tabletop", seed=11, replace=True)
    ]


@pytest.mark.parametrize(
    "body",
    [start_body(seed=-1), start_body(scene=None), {"robot_id": str(ROBOT_ID), "scene": "tabletop"}],
)
def test_post_with_a_malformed_body_is_a_bad_request(client: TestClient, manager: StubManager, body: dict) -> None:
    response = client.post("/api/simulation", json=body)

    assert response.status_code == 400
    assert response.json()["error_code"] == "bad_request"
    assert manager.started == []


@pytest.mark.parametrize(
    ("error", "status_code", "error_code"),
    [
        (
            SimulationNotSupportedError("Robot 'Arm' has no simulation."),
            400,
            "simulation_not_supported",
        ),
        (SimulationInvalidRequestError("Unknown scene 'garage'"), 400, "simulation_invalid_request"),
        (ResourceNotFoundError(ResourceType.ROBOT, ROBOT_ID), 404, "Robot_not_found"),
        (SimulationBusyError(project_id="p", robot_id="r", scene="s"), 409, "simulation_busy"),
        (SimulationNameInUseError("sim-a"), 409, "simulation_name_in_use"),
    ],
)
def test_post_maps_manager_errors(
    client: TestClient, manager: StubManager, error: Exception, status_code: int, error_code: str
) -> None:
    manager.start_error = error

    response = client.post("/api/simulation", json=start_body())

    assert response.status_code == status_code
    assert response.json()["error_code"] == error_code


def test_busy_names_the_running_simulation(client: TestClient, manager: StubManager) -> None:
    manager.start_error = SimulationBusyError(project_id="p-1", robot_id="r-1", scene="yahtzee")

    body = client.post("/api/simulation", json=start_body()).json()

    assert (body["project_id"], body["robot_id"], body["scene"]) == ("p-1", "r-1", "yahtzee")


def test_name_in_use_names_the_owner(client: TestClient, manager: StubManager) -> None:
    manager.start_error = SimulationNameInUseError("mujoco-so101-follow")

    assert client.post("/api/simulation", json=start_body()).json()["owner_name"] == "mujoco-so101-follow"


def test_delete_stops_and_returns_the_new_state(client: TestClient, manager: StubManager) -> None:
    manager.current = STARTING

    response = client.delete("/api/simulation")

    assert response.status_code == 200
    assert response.json()["status"] == "idle"
    assert manager.stopped == 1


def test_logs_returns_the_latest_lines(client: TestClient) -> None:
    response = client.get("/api/simulation/logs")

    assert response.status_code == 200
    assert response.json() == {"lines": ["loading scene", "ready"]}


async def test_events_stream_the_current_state_then_every_change(manager: StubManager) -> None:
    response = await simulation_events(manager)  # type: ignore[arg-type]
    events = response.body_iterator

    first = await anext(events)
    second = await anext(events)
    await events.aclose()  # type: ignore[attr-defined]

    assert json.loads(first["data"])["status"] == "idle"
    assert json.loads(second["data"])["status"] == "starting"


def test_catalog_describes_the_simulation_of_simulated_types() -> None:
    launch = SimpleNamespace(
        scenes=(SimpleNamespace(id="tabletop", display_name="Tabletop", description="A table"),),
        default_scene="tabletop",
        max_seed=4294967295,
        labels=("1 arm", "twin"),
    )

    response = robot_catalog._simulation_response(launch)

    assert response is not None
    assert response.model_dump() == {
        "scenes": [{"id": "tabletop", "display_name": "Tabletop", "description": "A table"}],
        "default_scene": "tabletop",
        "max_seed": 4294967295,
        "labels": ["1 arm", "twin"],
    }
    assert robot_catalog._simulation_response(None) is None

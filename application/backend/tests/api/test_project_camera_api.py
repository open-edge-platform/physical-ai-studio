# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Editing a camera added from a simulation: whether it stays its simulated robot's."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_camera_claim_registry, get_camera_service, get_robot_service, get_simulation_manager
from exceptions import ResourceNotFoundError, ResourceType
from main import app
from schemas.project_camera import IPCamera
from schemas.simulation import Simulation, SimulationStatus
from services.simulation_manager import _move_simulation_cameras

PROJECT_ID = uuid4()
A, B = uuid4(), uuid4()
A_URL, B_URL = "http://127.0.0.1:50001", "http://127.0.0.1:50002"


class FakeCameras:
    def __init__(self, camera: IPCamera) -> None:
        self.cameras = {camera.id: camera}
        self.session = SimpleNamespace(expire_all=lambda: None)

    async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> IPCamera:
        return self.cameras[camera_id]

    async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
        return list(self.cameras.values())

    async def update_camera(self, project_id: UUID, camera: IPCamera, commit: bool = True) -> IPCamera:
        self.cameras[camera.id] = camera
        return camera

    async def create_camera(self, project_id: UUID, camera: IPCamera) -> IPCamera:
        self.cameras[camera.id] = camera
        return camera


class FakeRobots:
    robots = {
        A: SimpleNamespace(id=A, payload=SimpleNamespace(http_url=A_URL)),
        B: SimpleNamespace(id=B, payload=SimpleNamespace(http_url=B_URL)),
    }

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
        if robot_id not in self.robots:
            raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))
        return self.robots[robot_id]


class StubManager:
    """Which robot's simulation Studio runs, and the lock Studio moves and adopts cameras under."""

    def __init__(self) -> None:
        self.current = Simulation()
        self.lock = asyncio.Lock()

    def simulated_by_studio(self, project_id: UUID, robot_id: UUID) -> bool:
        return self.current.status is SimulationStatus.RUNNING and self.current.robot_id == robot_id

    @contextlib.asynccontextmanager
    async def holding_addresses(self) -> AsyncIterator[Any]:
        async with self.lock:
            yield self.simulated_by_studio


@pytest.fixture
def manager() -> StubManager:
    return StubManager()


@pytest.fixture
def cameras(manager: StubManager) -> Iterator[FakeCameras]:
    url = f"{A_URL}/cameras/wrist/mjpeg"
    camera = IPCamera(
        id=uuid4(),
        name="A wrist",
        fingerprint={"url": url},
        hardware_name="A wrist",
        payload={"url": url, "simulation_robot_id": A},
    )
    fake = FakeCameras(camera)
    app.dependency_overrides[get_camera_service] = lambda: fake
    app.dependency_overrides[get_robot_service] = lambda: FakeRobots()
    app.dependency_overrides[get_camera_claim_registry] = lambda: SimpleNamespace(holder_of=lambda _: None)
    app.dependency_overrides[get_simulation_manager] = lambda: manager
    yield fake
    app.dependency_overrides.clear()


def edit_url(cameras: FakeCameras, url: str, form: IPCamera | None = None, name: str | None = None) -> IPCamera:
    """Save the camera form with a stream URL; the form sends back the rest of the camera as it opened it."""
    (camera,) = cameras.cameras.values()
    body = (form or camera).model_dump(mode="json")
    body["name"] = name or body["name"]
    body["fingerprint"] = {"url": url}
    body["payload"]["url"] = url
    response = TestClient(app).put(f"/api/projects/{PROJECT_ID}/cameras/{camera.id}", json=body)
    assert response.status_code == 200
    return cameras.cameras[camera.id]


async def test_a_camera_pointed_at_another_simulation_stays_there_when_its_robot_restarts(
    cameras: FakeCameras,
) -> None:
    edited = edit_url(cameras, f"{B_URL}/cameras/wrist/mjpeg")

    await _move_simulation_cameras(cameras, PROJECT_ID, A, "http://127.0.0.1:50003")  # type: ignore[arg-type]

    assert edited.payload.simulation_robot_id is None
    assert cameras.cameras[edited.id].payload.url == f"{B_URL}/cameras/wrist/mjpeg"


async def test_a_camera_pointed_at_another_stream_of_its_robot_still_follows_it(cameras: FakeCameras) -> None:
    edited = edit_url(cameras, f"{A_URL}/cameras/overview/mjpeg")

    await _move_simulation_cameras(cameras, PROJECT_ID, A, "http://127.0.0.1:50003")  # type: ignore[arg-type]

    assert edited.payload.simulation_robot_id == A
    assert cameras.cameras[edited.id].payload.url == "http://127.0.0.1:50003/cameras/overview/mjpeg"


def test_saving_a_camera_does_not_let_the_request_choose_its_robot(cameras: FakeCameras) -> None:
    (camera,) = cameras.cameras.values()
    body = camera.model_dump(mode="json")
    body["payload"]["simulation_robot_id"] = str(B)

    response = TestClient(app).put(f"/api/projects/{PROJECT_ID}/cameras/{camera.id}", json=body)

    assert response.status_code == 200
    assert cameras.cameras[camera.id].payload.simulation_robot_id == A


async def test_a_name_saved_from_a_form_opened_before_the_simulation_moved_keeps_its_new_url(
    cameras: FakeCameras, manager: StubManager
) -> None:
    (form,) = cameras.cameras.values()  # opened now, at A's address
    await _move_simulation_cameras(cameras, PROJECT_ID, A, "http://127.0.0.1:50003")  # type: ignore[arg-type]
    manager.current = Simulation(status=SimulationStatus.RUNNING, project_id=PROJECT_ID, robot_id=A)

    saved = edit_url(cameras, form.payload.url, form=form, name="Renamed")

    moved = "http://127.0.0.1:50003/cameras/wrist/mjpeg"
    assert (saved.name, saved.payload.url, saved.fingerprint, saved.payload.simulation_robot_id) == (
        "Renamed",
        moved,
        {"url": moved},
        A,
    )


def test_creating_a_camera_does_not_let_the_request_choose_its_robot(cameras: FakeCameras) -> None:
    url = f"{A_URL}/cameras/overview/mjpeg"
    body = {
        "name": "Mine now",
        "driver": "ipcam",
        "fingerprint": {"url": url},
        "hardware_name": None,
        "payload": {"url": url, "simulation_robot_id": str(A)},
    }

    response = TestClient(app).post(f"/api/projects/{PROJECT_ID}/cameras", json=body)

    assert response.status_code == 201
    (created,) = [camera for camera in cameras.cameras.values() if camera.name == "Mine now"]
    assert created.payload.simulation_robot_id is None


async def test_a_save_that_read_the_camera_before_it_was_adopted_keeps_the_adopting_robot(
    cameras: FakeCameras, manager: StubManager
) -> None:
    import api.project_camera as routes

    (camera,) = cameras.cameras.values()
    unowned = camera.model_copy(update={"payload": camera.payload.model_copy(update={"simulation_robot_id": None})})
    cameras.cameras[camera.id] = unowned
    read = cameras.get_camera_by_id
    reads = 0

    async def read_then_adopt(project_id: UUID, camera_id: UUID) -> IPCamera:
        # The save reads the camera; then an addition of A's cameras adopts it, under the lock.
        nonlocal reads
        reads += 1
        seen = await read(project_id, camera_id)
        if reads == 1:
            async with manager.holding_addresses():
                payload = seen.payload.model_copy(update={"simulation_robot_id": A})
                cameras.cameras[camera_id] = seen.model_copy(update={"payload": payload})
        return seen

    cameras.get_camera_by_id = read_then_adopt  # type: ignore[method-assign]
    form = unowned.model_copy(update={"name": "Renamed"})

    await routes.update_project_camera(
        PROJECT_ID,
        camera.id,
        cameras,  # type: ignore[arg-type]
        form,
        SimpleNamespace(holder_of=lambda _: None),  # type: ignore[arg-type]
        FakeRobots(),  # type: ignore[arg-type]
        manager,  # type: ignore[arg-type]
    )

    saved = cameras.cameras[camera.id]
    assert (saved.name, saved.payload.url, saved.payload.simulation_robot_id) == ("Renamed", unowned.payload.url, A)

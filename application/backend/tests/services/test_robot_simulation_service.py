# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobotSimulationService against a fake simulation HTTP API."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import BaseModel

import services.robot_simulation_service as module
from exceptions import SimulationInvalidRequestError, SimulationNotRunningError, SimulationNotSupportedError
from schemas.environment import CameraEnvironmentConfiguration, Environment
from schemas.project_camera import IPCamera
from schemas.simulation import Simulation, SimulationStatus
from services.robot_simulation_service import RobotSimulationService
from services.simulation_manager import _move_simulation_cameras

PROJECT_ID = uuid4()
ROBOT_ID = uuid4()
EXTERNAL = "http://127.0.0.1:8080"
STUDIO = "http://127.0.0.1:56209"


class Payload(BaseModel):
    name: str = "sim-a"
    http_url: str = EXTERNAL


def sim_api(base: str, *, service: str = "sim-a") -> dict[str, Any]:
    return {
        f"{base}/": {"service": service, "scene": "yahtzee", "viewer_url": f"{base[:-1]}1"},
        f"{base}/health": {"scene": "yahtzee"},
        f"{base}/cameras": [
            {
                "name": "wrist",
                "width": 640,
                "height": 480,
                "fps": 30,
                "failed": False,
                "source": "override",
                "stream_url": "/cameras/wrist/mjpeg",
                "snapshot_url": "/cameras/wrist/frame.jpg",
            },
            {"name": "broken", "failed": True},
        ],
    }


def transport(responses: dict[str, Any]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        return httpx.Response(200, json=responses[url]) if url in responses else httpx.Response(404)

    return httpx.MockTransport(handle)


@pytest.fixture
def robot(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    robot = SimpleNamespace(id=ROBOT_ID, name="Sim arm", type="Sim_Follower", payload=Payload(), others=[])

    class FakeRobotService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            return robot

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return [robot, *robot.others]

    monkeypatch.setattr(module, "RobotService", FakeRobotService)
    return robot


def service(
    responses: dict[str, Any], state: Simulation | None = None, simulated: bool = True
) -> RobotSimulationService:
    launch = SimpleNamespace(owner_name=lambda payload: payload.name)
    registry = SimpleNamespace(
        get_definition=lambda robot_type: SimpleNamespace(
            simulation=launch if simulated and robot_type.startswith("Sim") else None
        )
    )
    manager = SimpleNamespace(state=lambda: state or Simulation())
    session = SimpleNamespace(rollback=AsyncMock())
    return RobotSimulationService(session, registry, manager, transport=transport(responses))  # type: ignore[arg-type]


async def test_an_external_simulation_that_answers_under_the_robots_name(robot: SimpleNamespace) -> None:
    result = await service(sim_api(EXTERNAL)).get(PROJECT_ID, ROBOT_ID)

    assert result.running is True
    assert result.started_by == "external"
    assert result.http_url == EXTERNAL
    assert result.viewer_url == "http://127.0.0.1:8081"
    assert result.scene == "yahtzee"
    assert [(c.name, c.stream_url, c.snapshot_url) for c in result.cameras] == [
        ("wrist", f"{EXTERNAL}/cameras/wrist/mjpeg", f"{EXTERNAL}/cameras/wrist/frame.jpg")
    ]


async def test_studios_own_simulation_uses_the_managers_addresses(robot: SimpleNamespace) -> None:
    state = Simulation(
        status=SimulationStatus.RUNNING, robot_id=ROBOT_ID, http_url=STUDIO, viewer_url="http://127.0.0.1:56210"
    )

    result = await service(sim_api(STUDIO), state).get(PROJECT_ID, ROBOT_ID)

    assert result.started_by == "studio"
    assert result.viewer_url == "http://127.0.0.1:56210"
    assert result.cameras[0].stream_url == f"{STUDIO}/cameras/wrist/mjpeg"


async def test_a_simulation_studio_is_still_starting_is_not_running_yet(robot: SimpleNamespace) -> None:
    state = Simulation(status=SimulationStatus.STARTING, robot_id=ROBOT_ID)

    result = await service(sim_api(EXTERNAL), state).get(PROJECT_ID, ROBOT_ID)

    assert result.running is False


async def test_a_simulation_under_another_name_is_not_this_robots(robot: SimpleNamespace) -> None:
    result = await service(sim_api(EXTERNAL, service="someone-else")).get(PROJECT_ID, ROBOT_ID)

    assert result.running is False
    assert result.owner_name == "sim-a"


async def test_nothing_answering_is_not_running(robot: SimpleNamespace) -> None:
    result = await service({}).get(PROJECT_ID, ROBOT_ID)

    assert result.running is False
    assert result.cameras == []


async def test_a_robot_type_without_a_simulation_is_refused(robot: SimpleNamespace) -> None:
    with pytest.raises(SimulationNotSupportedError):
        await service({}, simulated=False).get(PROJECT_ID, ROBOT_ID)


@pytest.fixture
def environment_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store: dict[str, Any] = {
        "environment": Environment(
            id=uuid4(), name="Env", robots=[], cameras=[CameraEnvironmentConfiguration(camera_id=uuid4())]
        ),
        "created": [],
        "deleted": [],
        "existing": [],
        "updated": None,
        "fail_update": False,
    }

    class FakeEnvironmentRepository:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_by_id(self, environment_id: UUID) -> Any:
            return store["environment"]

    class FakeCameraService:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return store["existing"]

        async def create_camera(self, project_id: UUID, camera: IPCamera) -> IPCamera:
            store["created"].append(camera)
            return camera

        async def delete_camera(self, project_id: UUID, camera_id: UUID) -> None:
            store["deleted"].append(camera_id)

    class FakeEnvironmentService:
        def __init__(self, *args: Any) -> None:
            pass

        async def update_environment(self, project_id: UUID, environment: Environment) -> None:
            if store["fail_update"]:
                raise RuntimeError("database is gone")
            store["updated"] = environment

    monkeypatch.setattr(module, "ProjectEnvironmentRepository", FakeEnvironmentRepository)
    monkeypatch.setattr(module, "ProjectCameraService", FakeCameraService)
    monkeypatch.setattr(module, "EnvironmentService", FakeEnvironmentService)
    return store


async def test_adding_cameras_creates_ip_cameras_on_the_streams(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    environment = environment_store["environment"]

    created = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, environment.id, ROBOT_ID, ["wrist", "wrist"])

    assert [
        (c.name, c.hardware_name, c.payload.url, c.payload.width, c.payload.fps, c.fingerprint) for c in created
    ] == [
        (
            "Sim arm wrist",
            "Sim arm wrist",
            f"{EXTERNAL}/cameras/wrist/mjpeg",
            640,
            30,
            {"url": f"{EXTERNAL}/cameras/wrist/mjpeg"},
        )
    ]
    updated = environment_store["updated"]
    assert [c.camera_id for c in updated.cameras] == [environment.cameras[0].camera_id, created[0].id]


@pytest.mark.parametrize(("given", "expected"), [("  Wrist  ", "Wrist"), (" ", "Sim arm wrist")])
async def test_a_new_camera_takes_the_given_name_or_the_default_for_a_blank_one(
    robot: SimpleNamespace, environment_store: dict[str, Any], given: str, expected: str
) -> None:
    created = await service(sim_api(EXTERNAL)).add_cameras(
        PROJECT_ID, environment_store["environment"].id, ROBOT_ID, ["wrist"], {"wrist": given}
    )

    assert [(c.name, c.hardware_name) for c in created] == [(expected, expected)]


async def test_a_name_for_a_camera_that_is_not_added_is_refused(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    with pytest.raises(SimulationInvalidRequestError):
        await service(sim_api(EXTERNAL)).add_cameras(
            PROJECT_ID, environment_store["environment"].id, ROBOT_ID, ["wrist"], {"overview": "Top"}
        )


async def test_adding_a_camera_the_simulation_does_not_have_is_refused(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    with pytest.raises(SimulationInvalidRequestError, match="no camera named overview"):
        await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["overview"])

    assert environment_store["created"] == []


async def test_adding_cameras_needs_a_running_simulation(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    with pytest.raises(SimulationNotRunningError):
        await service({}).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])


async def test_moving_the_simulation_rewrites_only_its_camera_urls() -> None:
    def ipcam(url: str) -> IPCamera:
        return IPCamera(id=uuid4(), name=url, fingerprint={"url": url}, hardware_name=None, payload={"url": url})

    cameras = [
        ipcam(f"{EXTERNAL}/cameras/wrist/mjpeg"),
        ipcam("http://127.0.0.1:80800/cameras/wrist/mjpeg"),
        ipcam("rtsp://192.168.1.100:554/stream1"),
    ]
    updates: list[IPCamera] = []

    class FakeCameraService:
        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return cameras

        async def update_camera(self, project_id: UUID, camera: IPCamera) -> None:
            updates.append(camera)

    await _move_simulation_cameras(FakeCameraService(), PROJECT_ID, EXTERNAL, STUDIO)  # type: ignore[arg-type]

    assert [(c.id, c.payload.url, c.fingerprint) for c in updates] == [
        (cameras[0].id, f"{STUDIO}/cameras/wrist/mjpeg", {"url": f"{STUDIO}/cameras/wrist/mjpeg"})
    ]


def _ip_camera(url: str) -> IPCamera:
    return IPCamera(id=uuid4(), name="cam", fingerprint={"url": url}, hardware_name="cam", payload={"url": url})


async def test_a_camera_already_on_the_stream_is_reused(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    existing = _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg")
    environment_store["existing"] = [existing]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"], {"wrist": "New"})

    assert [(camera.id, camera.name) for camera in cameras] == [(existing.id, existing.name)]
    assert environment_store["created"] == []
    assert environment_store["updated"].cameras[-1].camera_id == existing.id


async def test_a_camera_already_in_the_environment_is_not_added_again(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    existing = _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg")
    environment_store["existing"] = [existing]
    environment_store["environment"] = environment_store["environment"].model_copy(
        update={"cameras": [CameraEnvironmentConfiguration(camera_id=existing.id)]}
    )

    await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert environment_store["created"] == []
    assert environment_store["updated"] is None


async def test_cameras_created_before_a_failure_are_removed(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    environment_store["fail_update"] = True

    with pytest.raises(RuntimeError):
        await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert environment_store["deleted"] == [camera.id for camera in environment_store["created"]]
    assert len(environment_store["deleted"]) == 1


async def test_listing_returns_the_running_simulations_of_simulated_robots(robot: SimpleNamespace) -> None:
    robot.others = [
        SimpleNamespace(id=uuid4(), name="Real arm", type="Real_Follower", payload=Payload(name="real")),
        SimpleNamespace(
            id=uuid4(), name="Idle sim", type="Sim_Follower", payload=Payload(name="idle", http_url=STUDIO)
        ),
    ]

    result = await service(sim_api(EXTERNAL)).list_running(PROJECT_ID)

    assert [(r.robot_id, r.robot_name, r.robot_type, r.started_by) for r in result] == [
        (ROBOT_ID, "Sim arm", "Sim_Follower", "external")
    ]


async def test_one_robot_that_cannot_be_probed_does_not_hide_the_others(robot: SimpleNamespace) -> None:
    robot.others = [
        SimpleNamespace(id=uuid4(), name="Broken", type="Sim_Follower", payload=Payload(http_url="http://[::1/")),
    ]

    result = await service(sim_api(EXTERNAL)).list_running(PROJECT_ID)

    assert [r.robot_name for r in result] == ["Sim arm"]

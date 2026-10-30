# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobotSimulationService against a fake simulation HTTP API."""

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import services.robot_simulation_service as module
from db.schema import Base, EnvironmentCameraDB, ProjectDB, ProjectEnvironmentDB
from exceptions import (
    SimulationCommandUnavailableError,
    SimulationInvalidRequestError,
    SimulationMovedError,
    SimulationNotRunningError,
    SimulationNotSupportedError,
)
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from schemas.environment import CameraEnvironmentConfiguration, Environment
from schemas.project_camera import IPCamera
from schemas.simulation import Simulation, SimulationStatus
from services.environment_service import EnvironmentService
from services.project_camera_service import ProjectCameraService
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


def fake_manager(state: Any) -> SimpleNamespace:
    """The parts of ``SimulationManager`` the service uses; *state* returns the manager's state."""

    @contextlib.asynccontextmanager
    async def updating_robot(project_id: UUID, robot_id: UUID):
        yield False

    return SimpleNamespace(state=state, updating_robot=updating_robot)


def service(
    responses: dict[str, Any], state: Simulation | None = None, simulated: bool = True, launch: Any = None
) -> RobotSimulationService:
    launch = launch or SimpleNamespace(owner_name=lambda payload: payload.name)
    registry = SimpleNamespace(
        get_definition=lambda robot_type: SimpleNamespace(
            simulation=launch if simulated and robot_type.startswith("Sim") else None
        )
    )
    manager = fake_manager(lambda: state or Simulation())
    session = SimpleNamespace(rollback=AsyncMock(), expire_all=lambda: None)
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
    # The browser gets them through Studio's own address; Studio keeps the simulation's.
    proxied = f"/api/projects/{PROJECT_ID}/robots/{ROBOT_ID}/simulation"
    assert result.viewer_proxy_url == f"{proxied}/viewer/"
    assert result.cameras[0].snapshot_proxy_url == f"{proxied}/http/cameras/wrist/frame.jpg"


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


def standalone_launch(*, scenes: tuple[str, ...] = ("yahtzee",)) -> SimpleNamespace:
    """A launch whose ``standalone_argv`` starts the robot's simulation at its payload's address."""

    def standalone_argv(scene: str, payload: Payload, seed: int | None = None) -> list[str]:
        if scene not in scenes:
            raise ValueError(f"Unknown scene {scene!r}")
        argv = ["sim", "start", "--scene", scene, f"--name={payload.name}", "--url", payload.http_url]
        return argv if seed is None else [*argv, "--seed", str(seed)]

    return SimpleNamespace(owner_name=lambda payload: payload.name, standalone_argv=standalone_argv)


async def test_the_command_starts_the_robots_simulation_at_its_address(robot: SimpleNamespace) -> None:
    robot.payload = Payload(name="my sim")

    result = await service({}, launch=standalone_launch()).command(PROJECT_ID, ROBOT_ID, "yahtzee", 7)

    assert result.argv == ["sim", "start", "--scene", "yahtzee", "--name=my sim", "--url", EXTERNAL, "--seed", "7"]
    assert result.command == f"sim start --scene yahtzee '--name=my sim' --url {EXTERNAL} --seed 7"


async def test_an_invalid_scene_for_the_command_is_refused(robot: SimpleNamespace) -> None:
    with pytest.raises(SimulationInvalidRequestError, match="Unknown scene 'fold'"):
        await service({}, launch=standalone_launch()).command(PROJECT_ID, ROBOT_ID, "fold", None)


@pytest.mark.parametrize(
    "launch",
    [
        # A plugin built against a physicalai-studio-plugin without `standalone_argv`.
        SimpleNamespace(owner_name=lambda payload: payload.name),
        # One whose launch has no `build_standalone_argv`.
        SimpleNamespace(owner_name=lambda payload: payload.name, standalone_argv=lambda *args: None),
    ],
)
async def test_a_plugin_without_a_command_offers_none(robot: SimpleNamespace, launch: SimpleNamespace) -> None:
    with pytest.raises(SimulationCommandUnavailableError):
        await service({}, launch=launch).command(PROJECT_ID, ROBOT_ID, "yahtzee", None)


async def test_a_robot_type_without_a_simulation_has_no_command(robot: SimpleNamespace) -> None:
    with pytest.raises(SimulationNotSupportedError):
        await service({}, simulated=False).command(PROJECT_ID, ROBOT_ID, "yahtzee", None)


@pytest.fixture
def environment_store(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    store: dict[str, Any] = {
        "environment": Environment(
            id=uuid4(), name="Env", robots=[], cameras=[CameraEnvironmentConfiguration(camera_id=uuid4())]
        ),
        "created": [],
        "deleted": [],
        "existing": [],
        "linked": [],
        "updated_cameras": [],
        "fail_update": False,
    }

    class FakeEnvironmentRepository:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_by_id(self, environment_id: UUID) -> Any:
            return store["environment"]

        async def add_cameras(
            self,
            environment_id: UUID,
            camera_ids: list[UUID],
            *,
            new_cameras: list[IPCamera] = (),  # type: ignore[assignment]
            changed_cameras: list[IPCamera] = (),  # type: ignore[assignment]
        ) -> None:
            # All or nothing, like the one transaction of the real repository.
            if store["fail_update"]:
                raise RuntimeError("database is gone")
            store["created"].extend(new_cameras)
            store["updated_cameras"].extend(changed_cameras)
            store["linked"].extend(camera_ids)

    class FakeCameraService:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return store["existing"]

        async def create_camera(self, project_id: UUID, camera: IPCamera) -> IPCamera:
            store["created"].append(camera)
            return camera

        async def update_camera(self, project_id: UUID, camera: IPCamera, commit: bool = True) -> IPCamera:
            store["updated_cameras"].append(camera)
            return camera

        async def delete_camera(self, project_id: UUID, camera_id: UUID) -> None:
            store["deleted"].append(camera_id)

    monkeypatch.setattr(module, "ProjectEnvironmentRepository", FakeEnvironmentRepository)
    monkeypatch.setattr(module, "ProjectCameraService", FakeCameraService)
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
    assert created[0].payload.simulation_robot_id == ROBOT_ID
    assert environment_store["linked"] == [created[0].id]


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


def _owned_camera(url: str, owner: UUID | None) -> IPCamera:
    return IPCamera(
        id=uuid4(),
        name=url,
        fingerprint={"url": url},
        hardware_name=None,
        payload={"url": url, "simulation_robot_id": owner},
    )


async def _move(cameras: list[IPCamera]) -> list[IPCamera]:
    updates: list[IPCamera] = []

    class FakeCameraService:
        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return cameras

        async def update_camera(self, project_id: UUID, camera: IPCamera, commit: bool = True) -> None:
            updates.append(camera)

    await _move_simulation_cameras(FakeCameraService(), PROJECT_ID, ROBOT_ID, STUDIO)  # type: ignore[arg-type]
    return updates


async def test_moving_the_simulation_rewrites_its_camera_urls_from_any_address() -> None:
    cameras = [
        _owned_camera(f"{EXTERNAL}/cameras/wrist/mjpeg", ROBOT_ID),
        _owned_camera("http://127.0.0.1:56100/cameras/overview/mjpeg", ROBOT_ID),
        _owned_camera(f"{STUDIO}/cameras/top/mjpeg", ROBOT_ID),  # already there
        _owned_camera("rtsp://192.168.1.100:554/stream1", ROBOT_ID),  # not a simulation stream
    ]

    updates = await _move(cameras)

    assert [(c.id, c.payload.url, c.fingerprint, c.payload.simulation_robot_id) for c in updates] == [
        (cameras[0].id, f"{STUDIO}/cameras/wrist/mjpeg", {"url": f"{STUDIO}/cameras/wrist/mjpeg"}, ROBOT_ID),
        (cameras[1].id, f"{STUDIO}/cameras/overview/mjpeg", {"url": f"{STUDIO}/cameras/overview/mjpeg"}, ROBOT_ID),
    ]


async def test_moving_the_simulation_leaves_another_robots_cameras_on_the_same_address() -> None:
    mine = _owned_camera(f"{EXTERNAL}/cameras/wrist/mjpeg", ROBOT_ID)
    theirs = _owned_camera(f"{EXTERNAL}/cameras/overview/mjpeg", uuid4())

    updates = await _move([mine, theirs])

    assert [c.id for c in updates] == [mine.id]


async def test_a_camera_of_no_robot_never_moves() -> None:
    assert await _move([_owned_camera(f"{EXTERNAL}/cameras/wrist/mjpeg", None)]) == []


def _ip_camera(url: str) -> IPCamera:
    return IPCamera(id=uuid4(), name="cam", fingerprint={"url": url}, hardware_name="cam", payload={"url": url})


async def test_a_camera_of_the_robot_already_on_the_stream_is_reused(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    existing = _owned_camera(f"{EXTERNAL}/cameras/wrist/mjpeg", ROBOT_ID)
    environment_store["existing"] = [existing]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"], {"wrist": "New"})

    assert [(camera.id, camera.name) for camera in cameras] == [(existing.id, existing.name)]
    assert environment_store["created"] == []
    assert environment_store["updated_cameras"] == []
    assert environment_store["linked"] == [existing.id]


async def test_a_camera_of_no_robot_on_the_stream_is_reused_and_becomes_the_robots(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    existing = _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg")
    environment_store["existing"] = [existing]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert [(camera.id, camera.payload.simulation_robot_id) for camera in cameras] == [(existing.id, ROBOT_ID)]
    assert [(c.id, c.payload.simulation_robot_id) for c in environment_store["updated_cameras"]] == [
        (existing.id, ROBOT_ID)
    ]
    assert environment_store["created"] == []
    assert environment_store["linked"] == [existing.id]


async def test_another_robots_camera_on_the_stream_is_not_reused(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    # The other robot's simulation ran on this port before, and moves its camera when it restarts.
    theirs = _owned_camera(f"{EXTERNAL}/cameras/wrist/mjpeg", uuid4())
    environment_store["existing"] = [theirs]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert [camera.id for camera in cameras] == [camera.id for camera in environment_store["created"]]
    assert cameras[0].id != theirs.id
    assert cameras[0].payload.simulation_robot_id == ROBOT_ID
    assert environment_store["updated_cameras"] == []
    assert environment_store["linked"] == [cameras[0].id]


async def test_a_failure_keeps_no_new_camera(robot: SimpleNamespace, environment_store: dict[str, Any]) -> None:
    environment_store["fail_update"] = True

    with pytest.raises(RuntimeError):
        await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert environment_store["created"] == []
    assert environment_store["linked"] == []


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


ENVIRONMENT_ID = uuid4()


@pytest.fixture
async def studio_db(tmp_path: Path):
    """A database with the project and an empty environment; yields its session factory."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add(ProjectDB(id=str(PROJECT_ID), name="Project"))
        session.add(ProjectEnvironmentDB(id=str(ENVIRONMENT_ID), project_id=str(PROJECT_ID), name="Rig"))
        await session.commit()
    yield sessions
    await engine.dispose()


def two_camera_api(on_cameras_probe: Any) -> httpx.MockTransport:
    """The simulation API with a wrist and an overview camera; awaits *on_cameras_probe* on each camera probe."""
    responses = sim_api(EXTERNAL)
    responses[f"{EXTERNAL}/cameras"] = [
        {**responses[f"{EXTERNAL}/cameras"][0], "name": name, "stream_url": f"/cameras/{name}/mjpeg"}
        for name in ("wrist", "overview")
    ]

    async def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{EXTERNAL}/cameras":
            await on_cameras_probe()
        return httpx.Response(200, json=responses[url]) if url in responses else httpx.Response(404)

    return httpx.MockTransport(handle)


async def add_with_database(session: Any, transport: httpx.MockTransport, name: str | list[str]) -> list[IPCamera]:
    launch = SimpleNamespace(owner_name=lambda payload: payload.name)
    registry = SimpleNamespace(get_definition=lambda robot_type: SimpleNamespace(simulation=launch))
    manager = fake_manager(Simulation)
    simulations = RobotSimulationService(session, registry, manager, transport=transport)  # type: ignore[arg-type]
    return await simulations.add_cameras(
        PROJECT_ID, ENVIRONMENT_ID, ROBOT_ID, [name] if isinstance(name, str) else name
    )


async def linked_cameras(sessions: Any) -> list[str]:
    async with sessions() as session:
        return sorted((await session.scalars(select(EnvironmentCameraDB.camera_id))).all())


async def test_concurrent_additions_to_an_environment_keep_each_others_cameras(
    robot: SimpleNamespace, studio_db: Any
) -> None:
    # The first two additions both read the environment before either adds its camera to it.
    both_probing = asyncio.Barrier(2)
    probes = 0

    async def wait_for_the_other() -> None:
        nonlocal probes
        probes += 1
        if probes <= 2:
            await asyncio.wait_for(both_probing.wait(), 5.0)

    transport = two_camera_api(wait_for_the_other)

    async def add(name: str) -> list[IPCamera]:
        async with studio_db() as session:
            return await add_with_database(session, transport, name)

    wrist, overview = await asyncio.gather(add("wrist"), add("overview"))
    await add("wrist")  # already in the environment

    assert await linked_cameras(studio_db) == sorted([str(wrist[0].id), str(overview[0].id)])


async def test_adding_cameras_keeps_a_change_made_to_the_environment_meanwhile(
    robot: SimpleNamespace, studio_db: Any
) -> None:
    registry = RobotCatalogRegistry()
    removed = _ip_camera("rtsp://192.168.1.100:554/stream1")
    async with studio_db() as session:
        await ProjectCameraService(session, registry).create_camera(PROJECT_ID, removed)
        environment = await ProjectEnvironmentRepository(session, PROJECT_ID, registry).get_by_id(ENVIRONMENT_ID)
        linked = [CameraEnvironmentConfiguration(camera_id=removed.id)]
        await EnvironmentService(session, registry).update_environment(
            PROJECT_ID, environment.model_copy(update={"cameras": linked})
        )

    async def remove_the_other_camera() -> None:
        # Someone saves the environment without that camera while the addition probes the simulation.
        async with studio_db() as session:
            environment = await ProjectEnvironmentRepository(session, PROJECT_ID, registry).get_by_id(ENVIRONMENT_ID)
            await EnvironmentService(session, registry).update_environment(
                PROJECT_ID, environment.model_copy(update={"cameras": []})
            )

    async with studio_db() as session:
        wrist = await add_with_database(session, two_camera_api(remove_the_other_camera), "wrist")

    assert await linked_cameras(studio_db) == [str(wrist[0].id)]


async def test_adding_cameras_is_refused_once_a_replacement_moved_the_simulation(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    unowned = _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg")
    environment_store["existing"] = [unowned]
    responses = sim_api(EXTERNAL)

    async def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == f"{EXTERNAL}/cameras":
            # While the old simulation is probed, a replacement starts on another port and stores its
            # address on the robot; its relocation skipped the camera, which had no robot yet.
            robot.payload = Payload(http_url=STUDIO)
        return httpx.Response(200, json=responses[url]) if url in responses else httpx.Response(404)

    simulations = RobotSimulationService(
        SimpleNamespace(rollback=AsyncMock(), expire_all=lambda: None),  # type: ignore[arg-type]
        SimpleNamespace(
            get_definition=lambda robot_type: SimpleNamespace(
                simulation=SimpleNamespace(owner_name=lambda payload: "sim-a")
            )
        ),  # type: ignore[arg-type]
        fake_manager(Simulation),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handle),
    )

    with pytest.raises(SimulationMovedError):
        await simulations.add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert environment_store["updated_cameras"] == []
    assert environment_store["created"] == []
    assert environment_store["linked"] == []


async def test_a_failed_addition_keeps_neither_an_adoption_nor_a_new_camera(
    robot: SimpleNamespace, studio_db: Any
) -> None:
    registry = RobotCatalogRegistry()
    unowned = _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg")
    async with studio_db() as session:
        await ProjectCameraService(session, registry).create_camera(PROJECT_ID, unowned)

    async def delete_the_environment() -> None:
        # The environment is deleted while the addition probes the simulation.
        async with studio_db() as session:
            await EnvironmentService(session, registry).delete_environment(PROJECT_ID, ENVIRONMENT_ID)

    async with studio_db() as session:
        with pytest.raises(IntegrityError):
            await add_with_database(session, two_camera_api(delete_the_environment), ["wrist", "overview"])

    async with studio_db() as session:
        stored = await ProjectCameraService(session, registry).get_camera_list(PROJECT_ID)
    assert [(camera.id, camera.payload.simulation_robot_id) for camera in stored] == [(unowned.id, None)]

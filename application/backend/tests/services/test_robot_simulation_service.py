# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobotSimulationService against a fake simulation HTTP API."""

import asyncio
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
from db.schema import Base, EnvironmentCameraDB, EnvironmentRobotDB, ProjectDB, ProjectEnvironmentDB, ProjectRobotDB
from exceptions import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    SimulationCommandUnavailableError,
    SimulationInvalidRequestError,
    SimulationNotRunningError,
    SimulationNotServingError,
    SimulationNotSupportedError,
)
from repositories.project_camera_repo import ProjectCameraRepository
from repositories.project_environment_repo import ProjectEnvironmentRepository
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService as RealRobotService
from schemas.environment import CameraEnvironmentConfiguration, Environment
from schemas.project_camera import IPCamera, IPCameraPayload, SimulationCamera, SimulationCameraPayload
from schemas.simulation import Simulation, SimulationStatus
from services.environment_service import EnvironmentService
from services.project_camera_service import ProjectCameraService
from services.robot_simulation_service import RobotSimulationService

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
    robot = SimpleNamespace(
        id=ROBOT_ID, name="Sim arm", type="Sim_Follower", payload=Payload(), others=[], deleted=False
    )

    class FakeRobotService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            if robot.deleted:
                raise ResourceNotFoundError(ResourceType.ROBOT, str(robot_id))
            return robot

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return [robot, *robot.others]

    monkeypatch.setattr(module, "RobotService", FakeRobotService)
    return robot


def fake_manager(state: Any) -> SimpleNamespace:
    """The parts of ``SimulationManager`` the service uses; *state* returns the manager's state."""
    return SimpleNamespace(state=state)


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
    session = SimpleNamespace(rollback=AsyncMock(), commit=AsyncMock(), expire_all=lambda: None)
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
        "existing": [],
        "linked": [],
        "fail_link": False,
    }

    class FakeEnvironmentRepository:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_by_id(self, environment_id: UUID) -> Any:
            return store["environment"]

        async def link_cameras(self, environment_id: UUID, camera_ids: list[UUID]) -> None:
            if store["fail_link"]:
                raise RuntimeError("database is gone")
            store["linked"].extend(camera_ids)

    class FakeCameraService:
        def __init__(self, *args: Any) -> None:
            pass

        async def get_camera_list(self, project_id: UUID) -> list[Any]:
            return store["existing"]

        async def get_camera_by_id(self, project_id: UUID, camera_id: UUID) -> Any:
            return next(camera for camera in [*store["existing"], *store["created"]] if camera.id == camera_id)

    class FakeCameraRepository:
        def __init__(self, *args: Any) -> None:
            pass

        async def add_if_absent(self, camera: Any) -> None:
            store["created"].append(camera)

    monkeypatch.setattr(module, "ProjectEnvironmentRepository", FakeEnvironmentRepository)
    monkeypatch.setattr(module, "ProjectCameraService", FakeCameraService)
    monkeypatch.setattr(module, "ProjectCameraRepository", FakeCameraRepository)
    return store


async def test_adding_cameras_creates_simulation_cameras(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    environment = environment_store["environment"]

    created = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, environment.id, ROBOT_ID, ["wrist", "wrist"])

    assert [(c.driver, c.name, c.hardware_name, c.payload, c.fingerprint) for c in created] == [
        (
            "simulation",
            "Sim arm wrist",
            "wrist",
            SimulationCameraPayload(robot_id=ROBOT_ID, camera="wrist"),
            {"robot_id": str(ROBOT_ID), "camera": "wrist"},
        )
    ]
    assert environment_store["created"] == created
    assert environment_store["linked"] == [created[0].id]


@pytest.mark.parametrize(("given", "expected"), [("  Wrist  ", "Wrist"), (" ", "Sim arm wrist")])
async def test_a_new_camera_takes_the_given_name_or_the_default_for_a_blank_one(
    robot: SimpleNamespace, environment_store: dict[str, Any], given: str, expected: str
) -> None:
    created = await service(sim_api(EXTERNAL)).add_cameras(
        PROJECT_ID, environment_store["environment"].id, ROBOT_ID, ["wrist"], {"wrist": given}
    )

    assert [c.name for c in created] == [expected]


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


def _simulation_camera(robot_id: UUID, camera: str, name: str = "cam") -> SimulationCamera:
    return SimulationCamera(
        id=uuid4(),
        name=name,
        fingerprint={"robot_id": str(robot_id), "camera": camera},
        hardware_name=camera,
        payload=SimulationCameraPayload(robot_id=robot_id, camera=camera),
    )


def _ip_camera(url: str) -> IPCamera:
    return IPCamera(id=uuid4(), name="cam", fingerprint={"url": url}, hardware_name="cam", payload={"url": url})


async def test_the_robots_camera_the_project_has_is_reused_under_its_own_name(
    robot: SimpleNamespace, environment_store: dict[str, Any]
) -> None:
    existing = _simulation_camera(ROBOT_ID, "wrist", "Old name")
    environment_store["existing"] = [existing]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"], {"wrist": "New"})

    assert [(camera.id, camera.name) for camera in cameras] == [(existing.id, "Old name")]
    assert environment_store["created"] == []
    assert environment_store["linked"] == [existing.id]


@pytest.mark.parametrize(
    "other",
    [
        _simulation_camera(uuid4(), "wrist"),  # another robot's camera of the same name
        _ip_camera(f"{EXTERNAL}/cameras/wrist/mjpeg"),  # an IP camera on the stream: left as it is
    ],
)
async def test_no_other_camera_is_reused(robot: SimpleNamespace, environment_store: dict[str, Any], other: Any) -> None:
    environment_store["existing"] = [other]

    cameras = await service(sim_api(EXTERNAL)).add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    assert cameras == environment_store["created"]
    assert cameras[0].id != other.id


async def test_a_failure_rolls_the_new_cameras_back(robot: SimpleNamespace, environment_store: dict[str, Any]) -> None:
    environment_store["fail_link"] = True
    simulations = service(sim_api(EXTERNAL))

    with pytest.raises(RuntimeError):
        await simulations.add_cameras(PROJECT_ID, uuid4(), ROBOT_ID, ["wrist"])

    simulations._session.rollback.assert_awaited_once()  # type: ignore[attr-defined]


async def test_a_simulation_camera_streams_from_where_its_simulation_runs_now(robot: SimpleNamespace) -> None:
    probes = 0
    responses = sim_api(STUDIO)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal probes
        url = str(request.url)
        probes += url == f"{STUDIO}/cameras"
        return httpx.Response(200, json=responses[url]) if url in responses else httpx.Response(404)

    state = Simulation(status=SimulationStatus.RUNNING, robot_id=ROBOT_ID, http_url=STUDIO)
    simulations = service({}, state)
    simulations._transport = httpx.MockTransport(handle)
    wrist = _simulation_camera(ROBOT_ID, "wrist", "Wrist")
    other = _ip_camera("rtsp://192.168.1.100:554/stream1")

    resolved = await simulations.resolve_cameras(PROJECT_ID, [wrist, other, wrist])

    assert resolved[1] is other
    assert resolved[0] == resolved[2]
    assert isinstance(resolved[0], IPCamera)
    assert (resolved[0].id, resolved[0].name, resolved[0].fingerprint, resolved[0].hardware_name) == (
        wrist.id,
        "Wrist",
        wrist.fingerprint,
        "wrist",
    )
    assert resolved[0].payload == IPCameraPayload(url=f"{STUDIO}/cameras/wrist/mjpeg", width=640, height=480, fps=30)
    assert probes == 1


async def test_a_simulation_camera_needs_its_simulation_running(robot: SimpleNamespace) -> None:
    with pytest.raises(SimulationNotRunningError, match="Start the simulation of 'Sim arm'"):
        await service({}).resolve_cameras(PROJECT_ID, [_simulation_camera(ROBOT_ID, "wrist")])


async def test_a_simulation_camera_the_simulation_no_longer_has_is_refused(robot: SimpleNamespace) -> None:
    with pytest.raises(SimulationNotServingError, match="no camera 'overview'"):
        await service(sim_api(EXTERNAL)).resolve_cameras(PROJECT_ID, [_simulation_camera(ROBOT_ID, "overview")])


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


async def add_with_database(
    session: Any, transport: httpx.MockTransport, name: str | list[str]
) -> list[SimulationCamera]:
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

    async def add(name: str) -> list[SimulationCamera]:
        async with studio_db() as session:
            return await add_with_database(session, transport, name)

    wrist, overview = await asyncio.gather(add("wrist"), add("overview"))
    await add("wrist")  # already in the environment

    assert await linked_cameras(studio_db) == sorted([str(wrist[0].id), str(overview[0].id)])


async def test_concurrent_additions_of_the_same_camera_make_one_camera(robot: SimpleNamespace, studio_db: Any) -> None:
    # Both additions find no camera of the robot before either creates one.
    both_probing = asyncio.Barrier(2)

    async def wait_for_the_other() -> None:
        await asyncio.wait_for(both_probing.wait(), 5.0)

    transport = two_camera_api(wait_for_the_other)

    async def add() -> list[SimulationCamera]:
        async with studio_db() as session:
            return await add_with_database(session, transport, "wrist")

    first, second = await asyncio.gather(add(), add())

    async with studio_db() as session:
        stored = await ProjectCameraService(session, RobotCatalogRegistry()).get_camera_list(PROJECT_ID)
    assert [camera.id for camera in stored] == [first[0].id] == [second[0].id]
    assert await linked_cameras(studio_db) == [str(first[0].id)]


async def test_a_robot_deleted_during_an_addition_keeps_no_camera(robot: SimpleNamespace, studio_db: Any) -> None:
    async def delete_the_robot() -> None:
        robot.deleted = True

    async with studio_db() as session:
        with pytest.raises(ResourceNotFoundError):
            await add_with_database(session, two_camera_api(delete_the_robot), "wrist")

    async with studio_db() as session:
        assert await ProjectCameraService(session, RobotCatalogRegistry()).get_camera_list(PROJECT_ID) == []
    assert await linked_cameras(studio_db) == []


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


async def test_a_failed_addition_keeps_no_new_camera(robot: SimpleNamespace, studio_db: Any) -> None:
    registry = RobotCatalogRegistry()
    existing = _ip_camera("rtsp://192.168.1.100:554/stream1")
    async with studio_db() as session:
        await ProjectCameraService(session, registry).create_camera(PROJECT_ID, existing)

    async def delete_the_environment() -> None:
        # The environment is deleted while the addition probes the simulation.
        async with studio_db() as session:
            await EnvironmentService(session, registry).delete_environment(PROJECT_ID, ENVIRONMENT_ID)

    async with studio_db() as session:
        with pytest.raises(IntegrityError):
            await add_with_database(session, two_camera_api(delete_the_environment), ["wrist", "overview"])

    async with studio_db() as session:
        stored = await ProjectCameraService(session, registry).get_camera_list(PROJECT_ID)
    assert [camera.id for camera in stored] == [existing.id]


async def test_deleting_a_robots_simulation_cameras_unlinks_them_and_keeps_the_others(studio_db: Any) -> None:
    registry = RobotCatalogRegistry()
    other_robot = uuid4()
    mine = _simulation_camera(ROBOT_ID, "wrist")
    theirs = _simulation_camera(other_robot, "wrist")
    plain = _ip_camera("rtsp://192.168.1.100:554/stream1")
    async with studio_db() as session:
        for camera in (mine, theirs, plain):
            await ProjectCameraService(session, registry).create_camera(PROJECT_ID, camera)
        await ProjectEnvironmentRepository(session, PROJECT_ID, registry).link_cameras(
            ENVIRONMENT_ID, [mine.id, theirs.id, plain.id]
        )
        await session.commit()

    async with studio_db() as session:
        await ProjectCameraRepository(session, str(PROJECT_ID)).delete_simulation_cameras(ROBOT_ID)
        await session.commit()

    async with studio_db() as session:
        stored = await ProjectCameraService(session, registry).get_camera_list(PROJECT_ID)
    assert sorted(str(camera.id) for camera in stored) == sorted([str(theirs.id), str(plain.id)])
    assert await linked_cameras(studio_db) == sorted([str(theirs.id), str(plain.id)])


async def _add_robot(sessions: Any, *, in_environment: bool) -> None:
    async with sessions() as session:
        session.add(
            ProjectRobotDB(
                id=str(ROBOT_ID),
                project_id=str(PROJECT_ID),
                name="Sim arm",
                type="SO101_Follower",
                payload={"connection_string": "/dev/ttyACM0", "serial_number": "arm"},
            )
        )
        if in_environment:
            session.add(EnvironmentRobotDB(environment_id=str(ENVIRONMENT_ID), robot_id=str(ROBOT_ID)))
        await session.commit()


async def _add_camera_of_the_robot(sessions: Any) -> SimulationCamera:
    camera = _simulation_camera(ROBOT_ID, "wrist")
    async with sessions() as session:
        registry = RobotCatalogRegistry()
        await ProjectCameraService(session, registry).create_camera(PROJECT_ID, camera)
        await ProjectEnvironmentRepository(session, PROJECT_ID, registry).link_cameras(ENVIRONMENT_ID, [camera.id])
        await session.commit()
    return camera


async def test_deleting_a_robot_deletes_its_simulation_cameras(studio_db: Any) -> None:
    await _add_robot(studio_db, in_environment=False)
    await _add_camera_of_the_robot(studio_db)

    async with studio_db() as session:
        await RealRobotService(session, RobotCatalogRegistry()).delete_robot(PROJECT_ID, ROBOT_ID)

    async with studio_db() as session:
        assert await ProjectCameraService(session, RobotCatalogRegistry()).get_camera_list(PROJECT_ID) == []
    assert await linked_cameras(studio_db) == []


async def test_a_robot_that_cannot_be_deleted_keeps_its_simulation_cameras(studio_db: Any) -> None:
    await _add_robot(studio_db, in_environment=True)
    camera = await _add_camera_of_the_robot(studio_db)

    async with studio_db() as session:
        with pytest.raises(ResourceInUseError):
            await RealRobotService(session, RobotCatalogRegistry()).delete_robot(PROJECT_ID, ROBOT_ID)

    async with studio_db() as session:
        stored = await ProjectCameraService(session, RobotCatalogRegistry()).get_camera_list(PROJECT_ID)
    assert [c.id for c in stored] == [camera.id]
    assert await linked_cameras(studio_db) == [str(camera.id)]

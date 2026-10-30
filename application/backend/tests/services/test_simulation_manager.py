# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SimulationManager against a fake simulation process that follows the launch contract."""

import asyncio
import contextlib
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db.schema import Base, EnvironmentRobotDB, ProjectDB, ProjectEnvironmentDB
from exceptions import (
    ResourceNotFoundError,
    ResourceType,
    SimulationBusyError,
    SimulationInvalidRequestError,
    SimulationNameInUseError,
    SimulationNotSupportedError,
)
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.environment import (
    CameraEnvironmentConfiguration,
    Environment,
    RobotEnvironmentConfiguration,
    TeleoperatorNone,
    TeleoperatorRobot,
)
from schemas.project_camera import IPCamera
from schemas.robot import RobotAdapter, UnavailableRobot
from schemas.simulation import Simulation, SimulationPhase, SimulationStatus, StartSimulationRequest
from services.simulation_manager import SimulationManager

# The fake simulation: prints the scenario's stderr lines and events, then either exits or waits for
# stdin to close. It can ignore end of input and SIGTERM to exercise the stop escalation.
FAKE_SIMULATION = textwrap.dedent(
    """
    import json, os, signal, sys, time

    scenario = json.loads(sys.argv[1])
    if scenario.get("ignore_term"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(scenario["pid_file"], "w") as f:
        f.write(str(os.getpid()))
    for line in scenario.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
    if scenario.get("grandchild_file"):
        # Like the runtime's owner: its own session, and it inherits stdout and stderr.
        import subprocess
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        with open(scenario["grandchild_file"], "w") as f:
            f.write(str(grandchild.pid))
    for line in scenario.get("stdout", []):
        print(line, flush=True)
    if scenario.get("long_line_bytes"):
        print("x" * scenario["long_line_bytes"], flush=True)
    for event in scenario.get("events", []):
        time.sleep(scenario.get("delay", 0.0))
        print(json.dumps(event), flush=True)
    if scenario.get("then") == "exit":
        sys.exit(scenario.get("exit_code", 0))
    sys.stdin.read()
    if scenario.get("ignore_eof"):
        while True:
            time.sleep(1)
    sys.exit(scenario.get("exit_code", 0))
    """
)

PHASES = [{"event": "phase", "phase": phase} for phase in ("fetch", "connect", "load", "cameras")]


def ready(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "event": "ready",
        "name": name,
        "pid": 4242,
        "http_url": "http://127.0.0.1:56209",
        "viewer_url": "http://127.0.0.1:56210",
        "cameras": ["wrist", "overview"],
        **extra,
    }


@dataclass(frozen=True)
class FakeScene:
    id: str
    display_name: str
    description: str = ""


@dataclass(frozen=True)
class FakeLaunch:
    """The part of ``physicalai_studio_plugin.SimulationLaunch`` the manager uses.

    The released plugin has no ``SimulationLaunch`` yet; ``test_the_real_launch_contract`` covers the
    real class when it is installed.
    """

    scenes: tuple[FakeScene, ...]
    default_scene: str
    payload_owner_name: Any
    build_argv: Any
    max_seed: int | None = None
    labels: tuple[str, ...] = ()

    def owner_name(self, payload: Any) -> str:
        return self.payload_owner_name(payload)

    def argv(self, scene: str, payload: Any, seed: int | None = None) -> list[str]:
        if scene not in {known.id for known in self.scenes}:
            raise ValueError(f"Unknown scene {scene!r}")
        if seed is not None and (self.max_seed is None or not 0 <= seed <= self.max_seed):
            raise ValueError(f"seed must be an integer from 0 to {self.max_seed}, got {seed!r}")
        return list(self.build_argv(scene=scene, owner_name=self.owner_name(payload), seed=seed))


class FakePayload(BaseModel):
    name: str = "sim-a"
    http_url: str = "http://127.0.0.1:8080"


@dataclass
class FakeRobots:
    robots: dict[UUID, Any] = field(default_factory=dict)
    http_urls: list[tuple[UUID, str]] = field(default_factory=list)
    fail_set_http_url: bool = False
    set_http_url_delay_s: float = 0.0

    def add(self, robot_type: str = "Sim_Follower", payload: BaseModel | None = None) -> UUID:
        robot_id = uuid4()
        self.robots[robot_id] = SimpleNamespace(
            id=robot_id, name="Sim", type=robot_type, payload=payload or FakePayload()
        )
        return robot_id

    async def get(self, project_id: UUID, robot_id: UUID) -> Any:
        if robot_id not in self.robots:
            raise ResourceNotFoundError(ResourceType.ROBOT, robot_id)
        return self.robots[robot_id]

    async def set_http_url(self, project_id: UUID, robot_id: UUID, http_url: str) -> None:
        await asyncio.sleep(self.set_http_url_delay_s)
        if self.fail_set_http_url:
            raise RuntimeError("database is gone")
        self.http_urls.append((robot_id, http_url))


class Harness:
    """A manager with one simulated robot type whose argv runs the fake simulation."""

    def __init__(self, tmp_path: Path, **manager_kwargs: Any) -> None:
        self.script = tmp_path / "fake_simulation.py"
        self.script.write_text(FAKE_SIMULATION)
        self.pid_file = tmp_path / "pid"
        self.scenario: dict[str, Any] = {}
        self.online = False
        self.robots = FakeRobots()
        self.launch: Any = FakeLaunch(
            scenes=(FakeScene("tabletop", "Tabletop"), FakeScene("floor", "Floor")),
            default_scene="tabletop",
            payload_owner_name=lambda payload: payload.name,
            build_argv=self._argv,
            max_seed=100,
        )
        probe = SimpleNamespace(is_online=self._is_online)
        self.definitions = {
            "Sim_Follower": SimpleNamespace(simulation=self.launch, probe=probe),
            "Real_Follower": SimpleNamespace(simulation=None, probe=probe),
            "Legacy_Follower": SimpleNamespace(probe=probe),
        }
        registry = SimpleNamespace(get_definition=self.definitions.get)
        self.manager = SimulationManager(registry, self.robots, **manager_kwargs)  # type: ignore[arg-type]
        self.project_id = uuid4()

    def _argv(self, *, scene: str, owner_name: str, seed: int | None) -> list[str]:
        events = [ready(owner_name) if event == "READY" else event for event in self.scenario.get("events", [])]
        scenario = {**self.scenario, "events": events, "pid_file": str(self.pid_file)}
        return [sys.executable, str(self.script), json.dumps(scenario)]

    async def _is_online(self, payload: BaseModel) -> bool:
        return self.online

    def request(self, robot_id: UUID, scene: str = "tabletop", **kwargs: Any) -> StartSimulationRequest:
        return StartSimulationRequest(project_id=self.project_id, robot_id=robot_id, scene=scene, **kwargs)

    def child_pid(self) -> int:
        return int(self.pid_file.read_text())

    async def wait_for(self, status: SimulationStatus, timeout_s: float = 10.0) -> Simulation:
        async def until() -> Simulation:
            async with self.manager.subscribe() as queue:
                while (state := await queue.get()).status is not status:
                    pass
                return state

        return await asyncio.wait_for(until(), timeout_s)


async def wait_until(condition: Any, timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("condition not met")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
async def harness(tmp_path: Path):
    harness = Harness(tmp_path, start_timeout_s=5.0, stop_eof_timeout_s=2.0, stop_term_timeout_s=2.0)
    yield harness
    await harness.manager.shutdown()


async def test_start_reports_phases_then_ready_and_stores_the_address(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {
        "events": [{"event": "phase", "phase": "fetch", "bytes": 10, "total": 100}, *PHASES, "READY"],
        "delay": 0.05,
    }
    seen: list[Simulation] = []

    async def collect() -> None:
        async with harness.manager.subscribe() as queue:
            while True:
                seen.append(await queue.get())

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)

    started = await harness.manager.start(harness.request(robot_id, seed=7))

    assert started.status is SimulationStatus.STARTING
    assert started.owner_name == "sim-a"
    assert started.scene == "tabletop"
    assert started.seed == 7
    assert started.robot_type == "Sim_Follower"
    assert started.started_at is not None
    # The address the robot had: the simulation took another port, as the dialog then says.
    assert started.robot_http_url == "http://127.0.0.1:8080"

    running = await harness.wait_for(SimulationStatus.RUNNING)
    collector.cancel()

    assert running.http_url == "http://127.0.0.1:56209"
    assert running.robot_http_url == "http://127.0.0.1:8080"
    assert running.viewer_url == "http://127.0.0.1:56210"
    assert running.viewer_proxy_url == f"/api/projects/{harness.project_id}/robots/{robot_id}/simulation/viewer/"
    assert running.cameras == ["wrist", "overview"]
    assert running.pid == 4242
    assert running.phase is None
    assert running.download is None
    assert harness.robots.http_urls == [(robot_id, "http://127.0.0.1:56209")]
    downloads = [state.download for state in seen if state.download is not None]
    assert downloads and downloads[0].bytes == 10 and downloads[0].total == 100
    phases = [state.phase for state in seen if state.phase is not None]
    assert SimulationPhase.CAMERAS in phases


async def test_stop_closes_stdin_and_ends_idle(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": [*PHASES, "READY"]}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)
    pid = harness.child_pid()

    stopping = await harness.manager.stop()

    assert stopping.status is SimulationStatus.STOPPING
    idle = await harness.wait_for(SimulationStatus.IDLE)
    assert idle == Simulation()
    assert not process_alive(pid)


async def test_stop_while_idle_is_a_no_op(harness: Harness) -> None:
    assert (await harness.manager.stop()) == Simulation()


async def test_error_event_fails_with_its_message_and_exit_code(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {
        "events": [PHASES[0], {"event": "error", "message": "Menagerie download failed"}],
        "then": "exit",
        "exit_code": 1,
    }
    await harness.manager.start(harness.request(robot_id))

    failed = await harness.wait_for(SimulationStatus.FAILED)
    await asyncio.sleep(0.2)
    failed = harness.manager.state()

    assert failed.error is not None
    assert failed.error.message == "Menagerie download failed"
    assert failed.error.phase is SimulationPhase.FETCH
    assert failed.error.exit_code == 1
    assert harness.robots.http_urls == []


async def test_exit_without_ready_fails_with_the_last_log_line(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"stderr": ["usage: start [-h]", "start: error: bad scene"], "then": "exit", "exit_code": 2}
    await harness.manager.start(harness.request(robot_id))

    failed = await harness.wait_for(SimulationStatus.FAILED)

    assert failed.error is not None
    assert failed.error.exit_code == 2
    assert "exited with code 2 before it was ready" in failed.error.message
    assert "start: error: bad scene" in failed.error.message
    assert harness.manager.logs() == ["usage: start [-h]", "start: error: bad scene"]


async def test_no_progress_for_the_start_timeout_fails_and_stops_the_process(tmp_path: Path) -> None:
    harness = Harness(tmp_path, start_timeout_s=0.5, stop_eof_timeout_s=2.0)
    robot_id = harness.robots.add()
    harness.scenario = {"events": PHASES[:2]}
    try:
        await harness.manager.start(harness.request(robot_id))

        failed = await harness.wait_for(SimulationStatus.FAILED)

        assert failed.error is not None
        assert "no progress in the connect phase" in failed.error.message
        await asyncio.sleep(0.5)
        assert harness.manager.state().error.exit_code == 0  # exited on end of input
    finally:
        await harness.manager.shutdown()


async def test_stop_terminates_a_simulation_that_ignores_end_of_input(tmp_path: Path) -> None:
    harness = Harness(tmp_path, stop_eof_timeout_s=0.3, stop_term_timeout_s=2.0)
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"], "ignore_eof": True}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)
    pid = harness.child_pid()

    await harness.manager.stop()

    await harness.wait_for(SimulationStatus.IDLE)
    assert not process_alive(pid)


async def test_stop_kills_a_simulation_that_ignores_sigterm(tmp_path: Path) -> None:
    harness = Harness(tmp_path, stop_eof_timeout_s=0.3, stop_term_timeout_s=0.3)
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"], "ignore_eof": True, "ignore_term": True}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)
    pid = harness.child_pid()

    await harness.manager.stop()

    await harness.wait_for(SimulationStatus.IDLE)
    assert not process_alive(pid)


async def test_shutdown_waits_until_the_simulation_exited(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)
    pid = harness.child_pid()

    await harness.manager.shutdown()

    assert harness.manager.state().status is SimulationStatus.IDLE
    assert not process_alive(pid)


async def test_a_second_start_is_busy_unless_it_replaces_the_first(harness: Harness) -> None:
    first = harness.robots.add()
    second = harness.robots.add(payload=FakePayload(name="sim-b"))
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(first))
    await harness.wait_for(SimulationStatus.RUNNING)
    first_pid = harness.child_pid()

    with pytest.raises(SimulationBusyError) as busy:
        await harness.manager.start(harness.request(second))
    assert busy.value.details["robot_id"] == str(first)

    replaced = await harness.manager.start(harness.request(second, scene="floor", replace=True))

    assert replaced.status is SimulationStatus.STARTING
    assert replaced.robot_id == second
    assert not process_alive(first_pid)
    running = await harness.wait_for(SimulationStatus.RUNNING)
    assert running.owner_name == "sim-b"
    assert running.scene == "floor"


async def test_a_simulation_that_exits_cleanly_while_running_goes_idle(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"], "then": "exit", "exit_code": 0}
    await harness.manager.start(harness.request(robot_id))

    await harness.wait_for(SimulationStatus.IDLE)


async def test_a_simulation_that_crashes_while_running_fails(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"], "stderr": ["Segmentation fault"], "then": "exit", "exit_code": 139}
    await harness.manager.start(harness.request(robot_id))

    failed = await harness.wait_for(SimulationStatus.FAILED)

    assert failed.error is not None
    assert failed.error.exit_code == 139
    assert "stopped unexpectedly with code 139" in failed.error.message
    assert "Segmentation fault" in failed.error.message


async def test_stop_clears_a_failed_simulation(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"then": "exit", "exit_code": 1}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.FAILED)

    assert (await harness.manager.stop()) == Simulation()


async def test_a_failed_simulation_does_not_block_the_next_start(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"then": "exit", "exit_code": 1}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.FAILED)

    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))

    assert (await harness.wait_for(SimulationStatus.RUNNING)).error is None


async def test_ready_under_another_name_fails(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": [ready("someone-else")]}
    await harness.manager.start(harness.request(robot_id))

    failed = await harness.wait_for(SimulationStatus.FAILED)

    assert failed.error is not None
    assert "'someone-else'" in failed.error.message
    assert harness.robots.http_urls == []


async def test_storing_the_address_failing_keeps_the_simulation_running(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.robots.fail_set_http_url = True
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))

    assert (await harness.wait_for(SimulationStatus.RUNNING)).http_url == "http://127.0.0.1:56209"


async def test_non_event_stdout_lines_go_to_the_log(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"stdout": ["plain text", '{"no": "event"}'], "events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)

    assert harness.manager.logs() == ["plain text", '{"no": "event"}']


async def test_a_simulation_already_running_outside_studio_is_refused(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.online = True

    with pytest.raises(SimulationNameInUseError) as error:
        await harness.manager.start(harness.request(robot_id))

    assert error.value.details == {"owner_name": "sim-a"}
    assert harness.manager.state() == Simulation()
    assert not harness.pid_file.exists()


@pytest.mark.parametrize(
    ("scene", "seed", "match"),
    [("garage", None, "Unknown scene 'garage'"), ("tabletop", 101, "seed must be an integer from 0 to 100")],
)
async def test_an_invalid_scene_or_seed_is_refused(harness: Harness, scene: str, seed: int | None, match: str) -> None:
    robot_id = harness.robots.add()

    with pytest.raises(SimulationInvalidRequestError, match=match):
        await harness.manager.start(harness.request(robot_id, scene=scene, seed=seed))

    assert not harness.pid_file.exists()


@pytest.mark.parametrize("robot_type", ["Real_Follower", "Legacy_Follower", "Removed_Follower"])
async def test_a_robot_type_without_a_simulation_is_refused(harness: Harness, robot_type: str) -> None:
    robot_id = harness.robots.add(robot_type=robot_type)

    with pytest.raises(SimulationNotSupportedError, match="has no simulation"):
        await harness.manager.start(harness.request(robot_id))


async def test_a_robot_whose_plugin_is_missing_is_refused(harness: Harness) -> None:
    robot_id = uuid4()
    harness.robots.robots[robot_id] = UnavailableRobot(id=robot_id, name="Gone", type="Gone_Follower", payload={})

    with pytest.raises(SimulationNotSupportedError, match="plugin for 'Gone_Follower' is not installed"):
        await harness.manager.start(harness.request(robot_id))


@pytest.mark.parametrize("http_url", ["http://192.168.1.20:8080", "http://sim-box.local:8080"])
async def test_a_robot_on_another_host_is_refused(harness: Harness, http_url: str) -> None:
    robot_id = harness.robots.add(payload=FakePayload(http_url=http_url))

    with pytest.raises(SimulationNotSupportedError, match="only starts simulations on this machine"):
        await harness.manager.start(harness.request(robot_id))


@pytest.mark.parametrize("http_url", ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080"])
async def test_a_robot_on_this_machine_is_accepted(harness: Harness, http_url: str) -> None:
    robot_id = harness.robots.add(payload=FakePayload(http_url=http_url))
    harness.scenario = {"events": ["READY"]}

    await harness.manager.start(harness.request(robot_id))

    await harness.wait_for(SimulationStatus.RUNNING)


async def test_an_unknown_robot_is_not_found(harness: Harness) -> None:
    with pytest.raises(ResourceNotFoundError):
        await harness.manager.start(harness.request(uuid4()))


async def test_a_slow_subscriber_only_sees_the_latest_state(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": [*PHASES, "READY"]}

    async with harness.manager.subscribe() as queue:
        await harness.manager.start(harness.request(robot_id))
        await harness.wait_for(SimulationStatus.RUNNING)

        assert queue.qsize() == 1
        assert (await queue.get()).status is SimulationStatus.RUNNING


async def test_the_real_launch_contract(harness: Harness) -> None:
    simulation = pytest.importorskip("physicalai_studio_plugin.simulation")
    launch = harness.launch
    harness.definitions["Sim_Follower"].simulation = simulation.SimulationLaunch(
        scenes=tuple(simulation.SimulationScene(scene.id, scene.display_name) for scene in launch.scenes),
        default_scene=launch.default_scene,
        payload_owner_name=launch.payload_owner_name,
        build_argv=launch.build_argv,
        max_seed=launch.max_seed,
    )
    robot_id = harness.robots.add()
    harness.scenario = {"events": [*PHASES, "READY"]}

    with pytest.raises(SimulationInvalidRequestError, match="Unknown scene 'garage'"):
        await harness.manager.start(harness.request(robot_id, scene="garage"))
    await harness.manager.start(harness.request(robot_id, seed=3))

    assert (await harness.wait_for(SimulationStatus.RUNNING)).owner_name == "sim-a"


async def test_stop_does_not_wait_for_a_process_the_simulation_left_holding_its_output(tmp_path: Path) -> None:
    harness = Harness(tmp_path, stop_eof_timeout_s=5.0)
    grandchild_file = tmp_path / "grandchild"
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY"], "grandchild_file": str(grandchild_file)}
    try:
        await harness.manager.start(harness.request(robot_id))
        await harness.wait_for(SimulationStatus.RUNNING)
        pid = harness.child_pid()

        await harness.manager.stop()

        await harness.wait_for(SimulationStatus.IDLE, timeout_s=3.0)
        assert not process_alive(pid)
    finally:
        await harness.manager.shutdown()
        if grandchild_file.exists():
            os.kill(int(grandchild_file.read_text()), 9)


async def test_stop_while_starting_ends_idle(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    robot_id = harness.robots.add()
    harness.scenario = {"events": [*PHASES, "READY"], "delay": 1.0}
    try:
        await harness.manager.start(harness.request(robot_id))

        assert (await harness.manager.stop()).status is SimulationStatus.STOPPING
        await harness.wait_for(SimulationStatus.IDLE)
    finally:
        await harness.manager.shutdown()


async def test_an_error_event_while_running_fails_and_stops_the_process(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"events": ["READY", {"event": "error", "message": "physics diverged"}]}
    await harness.manager.start(harness.request(robot_id))

    failed = await harness.wait_for(SimulationStatus.FAILED)

    assert failed.error is not None and failed.error.message == "physics diverged"
    await wait_until(lambda: harness.manager.state().error.exit_code == 0)


async def test_stopping_a_failed_simulation_that_is_still_exiting_reports_stopping(tmp_path: Path) -> None:
    harness = Harness(tmp_path, stop_eof_timeout_s=0.5, stop_term_timeout_s=2.0)
    robot_id = harness.robots.add()
    harness.scenario = {"events": [{"event": "error", "message": "boom"}], "ignore_eof": True}
    try:
        await harness.manager.start(harness.request(robot_id))
        await harness.wait_for(SimulationStatus.FAILED)

        stopping = await harness.manager.stop()

        assert stopping.status is SimulationStatus.STOPPING
        assert stopping.error is None
        await harness.wait_for(SimulationStatus.IDLE)
    finally:
        await harness.manager.shutdown()


async def test_an_over_long_output_line_is_dropped_and_reading_goes_on(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.scenario = {"long_line_bytes": 2 << 20, "events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))

    await harness.wait_for(SimulationStatus.RUNNING)

    assert any("dropped an output line" in line for line in harness.manager.logs())


async def test_a_slow_address_write_does_not_delay_stopping(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.robots.set_http_url_delay_s = 30.0
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))
    await asyncio.sleep(0.5)  # ready is in, the address is being stored

    await harness.manager.stop()

    await harness.wait_for(SimulationStatus.IDLE, timeout_s=3.0)


async def test_running_is_reported_once_the_address_is_stored(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.robots.set_http_url_delay_s = 0.3
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))

    await harness.wait_for(SimulationStatus.RUNNING)

    assert harness.robots.http_urls == [(robot_id, "http://127.0.0.1:56209")]


async def test_a_stop_while_the_address_is_stored_is_not_undone(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.robots.set_http_url_delay_s = 0.5
    harness.scenario = {"events": ["READY"]}
    await harness.manager.start(harness.request(robot_id))
    await asyncio.sleep(0.2)

    await harness.manager.stop()
    await asyncio.sleep(0.6)

    assert harness.manager.state().status is not SimulationStatus.RUNNING


async def test_a_replaced_simulations_address_write_lands_before_the_new_ones(harness: Harness) -> None:
    robot_id = harness.robots.add()
    harness.robots.set_http_url_delay_s = 1.0
    harness.scenario = {"events": [ready("sim-a", http_url="http://127.0.0.1:50001")]}
    await harness.manager.start(harness.request(robot_id))
    await asyncio.sleep(0.5)  # ready is in, the first address is being stored

    harness.robots.set_http_url_delay_s = 0.0
    harness.scenario = {"events": [ready("sim-a", http_url="http://127.0.0.1:50002")]}
    await harness.manager.start(harness.request(robot_id, replace=True))
    await harness.wait_for(SimulationStatus.RUNNING)
    await asyncio.sleep(1.0)  # past the first write's delay

    assert harness.robots.http_urls == [(robot_id, "http://127.0.0.1:50001"), (robot_id, "http://127.0.0.1:50002")]


async def test_a_robot_update_sees_whether_studio_simulates_the_robot(harness: Harness) -> None:
    robot_id, other_id = harness.robots.add(), harness.robots.add()
    harness.scenario = {"events": ["READY"]}

    async def simulated(robot: UUID) -> bool:
        async with harness.manager.updating_robot(harness.project_id, robot) as simulated:
            return simulated

    assert await simulated(robot_id) is False
    await harness.manager.start(harness.request(robot_id))
    await harness.wait_for(SimulationStatus.RUNNING)
    assert [await simulated(robot_id), await simulated(other_id)] == [True, False]
    await harness.manager.stop()
    await harness.wait_for(SimulationStatus.IDLE)
    assert await simulated(robot_id) is False


async def test_a_robot_that_allows_remote_simulations_is_refused(harness: Harness) -> None:
    class RemotePayload(FakePayload):
        allow_remote: bool = True

    robot_id = harness.robots.add(payload=RemotePayload())

    with pytest.raises(SimulationNotSupportedError, match="allows remote simulations"):
        await harness.manager.start(harness.request(robot_id))


class _PayloadWithoutAddress(BaseModel):
    name: str = "sim-a"


@pytest.mark.parametrize(
    ("robot", "updated_url"),
    [
        (SimpleNamespace(type="Sim_Follower", payload=FakePayload(http_url="http://127.0.0.1:8080")), "new"),
        (SimpleNamespace(type="Sim_Follower", payload=FakePayload(http_url="http://127.0.0.1:9")), None),
        (SimpleNamespace(type="Sim_Follower", payload=_PayloadWithoutAddress()), None),
        (UnavailableRobot(id=uuid4(), name="Gone", type="Gone_Follower", payload={}), None),
    ],
    ids=["different", "same", "no-field", "unavailable"],
)
async def test_the_database_store_writes_the_address_only_when_it_changes(
    monkeypatch: pytest.MonkeyPatch, robot: Any, updated_url: str | None
) -> None:
    import services.simulation_manager as module

    if not isinstance(robot, UnavailableRobot):
        robot = SimpleNamespace(**vars(robot), model_copy=lambda update: SimpleNamespace(**update))
    updates: list[Any] = []

    class FakeService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            return robot

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return []

        async def update_robot(self, project_id: UUID, updated: Any, commit: bool = True) -> None:
            updates.append(updated)

    @contextlib.asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(commit=_nothing)

    monkeypatch.setattr(module, "RobotService", FakeService)
    monkeypatch.setattr(module, "ProjectCameraService", lambda *_args: SimpleNamespace(get_camera_list=_no_cameras))
    monkeypatch.setattr(module, "get_async_db_session_ctx", fake_session)
    monkeypatch.setattr(module, "EnvironmentService", lambda *_args: SimpleNamespace(get_environment_list=_no_list))
    new_url = "http://127.0.0.1:9"

    await module.DatabaseSimulationRobots(SimpleNamespace()).set_http_url(uuid4(), uuid4(), new_url)  # type: ignore[arg-type]

    if updated_url is None:
        assert updates == []
    else:
        assert [update.payload.http_url for update in updates] == [new_url]


async def _no_cameras(project_id: UUID) -> list[Any]:
    return []


async def _nothing() -> None:
    pass


async def _no_list(project_id: UUID) -> list[Any]:
    return []


class _StoredRobot(BaseModel):
    id: UUID
    name: str = "Sim"
    type: str = "Sim_Follower"
    payload: FakePayload


async def test_the_database_store_moves_only_the_started_robots_cameras_after_sequential_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import services.simulation_manager as module

    shared = "http://127.0.0.1:8080"
    a, b = uuid4(), uuid4()
    robots = {
        a: _StoredRobot(id=a, payload=FakePayload(name="sim-a", http_url=shared)),
        b: _StoredRobot(id=b, payload=FakePayload(name="sim-b", http_url=shared)),
    }

    def camera(stream: str, owner: UUID | None) -> IPCamera:
        url = f"{shared}/cameras/{stream}/mjpeg"
        return IPCamera(
            id=uuid4(),
            name=stream,
            fingerprint={"url": url},
            hardware_name=stream,
            payload={"url": url, "simulation_robot_id": owner},
        )

    cameras = {cam.name: cam for cam in (camera("wrist", a), camera("unowned", None))}

    class FakeRobotService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            return robots[robot_id]

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return list(robots.values())

        async def update_robot(self, project_id: UUID, updated: Any, commit: bool = True) -> None:
            robots[updated.id] = updated

    class FakeCameraService:
        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return list(cameras.values())

        async def update_camera(self, project_id: UUID, updated: IPCamera, commit: bool = True) -> None:
            cameras[updated.name] = updated

    @contextlib.asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(commit=_nothing)

    monkeypatch.setattr(module, "RobotService", FakeRobotService)
    monkeypatch.setattr(module, "ProjectCameraService", lambda *_args: FakeCameraService())
    monkeypatch.setattr(module, "get_async_db_session_ctx", fake_session)
    monkeypatch.setattr(module, "EnvironmentService", lambda *_args: SimpleNamespace(get_environment_list=_no_list))
    store = module.DatabaseSimulationRobots(SimpleNamespace())  # type: ignore[arg-type]

    await store.set_http_url(uuid4(), a, "http://127.0.0.1:50001")
    await store.set_http_url(uuid4(), b, "http://127.0.0.1:50002")

    assert {name: (cam.payload.url, cam.payload.simulation_robot_id) for name, cam in cameras.items()} == {
        "wrist": ("http://127.0.0.1:50001/cameras/wrist/mjpeg", a),
        "unowned": (f"{shared}/cameras/unowned/mjpeg", None),
    }
    assert [robots[a].payload.http_url, robots[b].payload.http_url] == [
        "http://127.0.0.1:50001",
        "http://127.0.0.1:50002",
    ]


async def test_a_start_on_the_robots_own_port_moves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The simulation reuses the robot's port when it is free: cameras, leaders and payload stay put."""
    import services.simulation_manager as module

    address = "http://127.0.0.1:8765"
    robot_id, leader_id = uuid4(), uuid4()
    robots: dict[UUID, Any] = {
        robot_id: _StoredRobot(id=robot_id, payload=FakePayload(name="sim-a", http_url=address)),
        leader_id: _StoredLeader(id=leader_id, payload=_StoredLeaderPayload(http_port=8765)),
    }
    url = f"{address}/cameras/wrist/mjpeg"
    camera = IPCamera(
        id=uuid4(),
        name="wrist",
        fingerprint={"url": url},
        hardware_name="wrist",
        payload={"url": url, "simulation_robot_id": robot_id},
    )
    environment = Environment(
        id=uuid4(),
        name="Env",
        robots=[RobotEnvironmentConfiguration(robot_id=robot_id, tele_operator=TeleoperatorRobot(robot_id=leader_id))],
    )
    updates: list[Any] = []

    class FakeRobotService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            return robots[robot_id]

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return list(robots.values())

        async def update_robot(self, project_id: UUID, updated: Any, commit: bool = True) -> None:
            updates.append(updated)

    class FakeCameraService:
        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return [camera]

        async def update_camera(self, project_id: UUID, updated: IPCamera, commit: bool = True) -> None:
            updates.append(updated)

    async def environments(project_id: UUID) -> list[Environment]:
        return [environment]

    @contextlib.asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(commit=_nothing)

    roles = {"Sim_Follower": "follower", "Sim_Leader": "leader"}
    registry = SimpleNamespace(get_definition=lambda robot_type: SimpleNamespace(role=roles[robot_type]))
    monkeypatch.setattr(module, "RobotService", FakeRobotService)
    monkeypatch.setattr(module, "ProjectCameraService", lambda *_args: FakeCameraService())
    monkeypatch.setattr(module, "get_async_db_session_ctx", fake_session)
    monkeypatch.setattr(module, "EnvironmentService", lambda *_args: SimpleNamespace(get_environment_list=environments))

    await module.DatabaseSimulationRobots(registry).set_http_url(uuid4(), robot_id, address)  # type: ignore[arg-type]

    assert updates == []


@pytest.mark.parametrize(
    ("environments", "camera_address", "moves"),
    [
        # The tester's database: cameras saved by hand on the robot's stream, in its only environment.
        ([["robot"]], "http://127.0.0.1:8765", True),
        ([["robot"], ["robot", "leader"]], "http://127.0.0.1:8765", True),
        # Next to another simulated robot, either robot could claim the camera.
        ([["robot", "other"]], "http://127.0.0.1:8765", False),
        ([["robot"], ["other"]], "http://127.0.0.1:8765", False),
        # In no environment: nothing says whose it is.
        ([], "http://127.0.0.1:8765", False),
        # On another address than the robot's.
        ([["robot"]], "http://127.0.0.1:9000", False),
    ],
    ids=["own-environment", "with-a-leader", "next-to-another-sim", "also-elsewhere", "no-environment", "elsewhere"],
)
async def test_a_fallback_start_takes_the_cameras_saved_by_hand_in_the_robots_environment(
    monkeypatch: pytest.MonkeyPatch, environments: list[list[str]], camera_address: str, moves: bool
) -> None:
    """Studio's start fell back to a free port: the robot's hand-saved cameras must follow it."""
    import services.simulation_manager as module

    ids = {"robot": uuid4(), "other": uuid4(), "leader": uuid4()}
    robots: dict[UUID, Any] = {
        ids["robot"]: _StoredRobot(id=ids["robot"], payload=FakePayload(name="fr3", http_url="http://127.0.0.1:8765")),
        ids["other"]: _StoredRobot(id=ids["other"], payload=FakePayload(name="b", http_url="http://127.0.0.1:8765")),
        ids["leader"]: _StoredLeader(id=ids["leader"], payload=_StoredLeaderPayload(http_port=9999)),
    }
    cameras = {
        name: IPCamera(
            id=uuid4(),
            name=name,
            fingerprint={"url": f"{camera_address}/cameras/{name}/mjpeg"},
            hardware_name=name,
            payload={"url": f"{camera_address}/cameras/{name}/mjpeg"},
        )
        for name in ("wrist", "overview")
    }
    project_environments = [
        Environment(
            id=uuid4(),
            name="franka",
            robots=[
                RobotEnvironmentConfiguration(robot_id=ids[member], tele_operator=TeleoperatorNone())
                for member in members
            ],
            cameras=[CameraEnvironmentConfiguration(camera_id=camera.id) for camera in cameras.values()],
        )
        for members in environments
    ]

    class FakeRobotService:
        def __init__(self, session: Any, registry: Any) -> None:
            pass

        async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
            return robots[robot_id]

        async def get_robot_list(self, project_id: UUID) -> list[Any]:
            return list(robots.values())

        async def update_robot(self, project_id: UUID, updated: Any, commit: bool = True) -> None:
            robots[updated.id] = updated

    class FakeCameraService:
        async def get_camera_list(self, project_id: UUID) -> list[IPCamera]:
            return list(cameras.values())

        async def update_camera(self, project_id: UUID, updated: IPCamera, commit: bool = True) -> None:
            cameras[updated.name] = updated

    async def environment_list(project_id: UUID) -> list[Environment]:
        return project_environments

    @contextlib.asynccontextmanager
    async def fake_session():
        yield SimpleNamespace(commit=_nothing)

    roles = {"Sim_Follower": "follower", "Sim_Leader": "leader"}
    registry = SimpleNamespace(get_definition=lambda robot_type: SimpleNamespace(role=roles[robot_type]))
    monkeypatch.setattr(module, "RobotService", FakeRobotService)
    monkeypatch.setattr(module, "ProjectCameraService", lambda *_args: FakeCameraService())
    monkeypatch.setattr(module, "get_async_db_session_ctx", fake_session)
    monkeypatch.setattr(
        module, "EnvironmentService", lambda *_args: SimpleNamespace(get_environment_list=environment_list)
    )

    fallback = "http://127.0.0.1:58153"
    await module.DatabaseSimulationRobots(registry).set_http_url(uuid4(), ids["robot"], fallback)  # type: ignore[arg-type]

    assert robots[ids["robot"]].payload.http_url == fallback
    expected = (fallback, ids["robot"]) if moves else (camera_address, None)
    for name, camera in cameras.items():
        assert (camera.payload.url, camera.payload.simulation_robot_id) == (
            f"{expected[0]}/cameras/{name}/mjpeg",
            expected[1],
        )


class _StoredLeaderPayload(BaseModel):
    http_port: int


class _StoredLeader(BaseModel):
    id: UUID
    name: str = "Leader"
    type: str = "Sim_Leader"
    payload: _StoredLeaderPayload


class _HardwareLeaderPayload(BaseModel):
    port: str = "/dev/ttyACM0"


class _HardwareLeader(BaseModel):
    id: UUID
    name: str = "Arm"
    type: str = "Real_Leader"
    payload: _HardwareLeaderPayload = _HardwareLeaderPayload()


class LeaderWorld:
    """A simulated follower and leaders in environments, behind ``DatabaseSimulationRobots``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import services.simulation_manager as module

        self.project_id, self.other_project_id = uuid4(), uuid4()
        self.follower = uuid4()
        self.robots: dict[UUID, Any] = {
            self.follower: _StoredRobot(id=self.follower, payload=FakePayload(http_url="http://127.0.0.1:8080"))
        }
        self.updates: list[UUID] = []
        self.environments: dict[UUID, list[Environment]] = {self.project_id: [], self.other_project_id: []}
        robots, environments, updates = self.robots, self.environments, self.updates

        class FakeRobotService:
            def __init__(self, session: Any, registry: Any) -> None:
                pass

            async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> Any:
                return robots[robot_id]

            async def get_robot_list(self, project_id: UUID) -> list[Any]:
                return list(robots.values())

            async def update_robot(self, project_id: UUID, updated: Any, commit: bool = True) -> None:
                robots[updated.id] = updated
                updates.append(updated.id)

        class FakeEnvironmentService:
            def __init__(self, session: Any, registry: Any) -> None:
                pass

            async def get_environment_list(self, project_id: UUID) -> list[Environment]:
                return environments[project_id]

        @contextlib.asynccontextmanager
        async def fake_session():
            yield SimpleNamespace(commit=_nothing)

        roles = {
            "Sim_Follower": "follower",
            "Sim_Leader": "leader",
            "Real_Follower": "follower",
            "Real_Leader": "leader",
        }
        registry = SimpleNamespace(
            get_definition=lambda robot_type: SimpleNamespace(
                role=roles[robot_type], simulation=object() if robot_type == "Sim_Follower" else None
            )
        )
        monkeypatch.setattr(module, "RobotService", FakeRobotService)
        monkeypatch.setattr(module, "EnvironmentService", FakeEnvironmentService)
        monkeypatch.setattr(module, "ProjectCameraService", lambda *_args: SimpleNamespace(get_camera_list=_no_cameras))
        monkeypatch.setattr(module, "get_async_db_session_ctx", fake_session)
        self.store = module.DatabaseSimulationRobots(registry)  # type: ignore[arg-type]
        self.manager = SimulationManager(registry, self.store)  # type: ignore[arg-type]

    def leader(self, http_port: int, *, robot_type: str = "Sim_Leader") -> UUID:
        leader_id = uuid4()
        self.robots[leader_id] = _StoredLeader(
            id=leader_id, type=robot_type, payload=_StoredLeaderPayload(http_port=http_port)
        )
        return leader_id

    def environment(
        self, *robot_ids: UUID, teleoperated_by: UUID | None = None, project_id: UUID | None = None
    ) -> None:
        tele_operator = TeleoperatorRobot(robot_id=teleoperated_by) if teleoperated_by else TeleoperatorNone()
        robots = [RobotEnvironmentConfiguration(robot_id=robot_ids[0], tele_operator=tele_operator)]
        robots += [
            RobotEnvironmentConfiguration(robot_id=other, tele_operator=TeleoperatorNone()) for other in robot_ids[1:]
        ]
        self.environments[project_id or self.project_id].append(Environment(id=uuid4(), name="Env", robots=robots))

    def simulated_follower(self, http_url: str, *, robot_type: str = "Sim_Follower") -> UUID:
        follower_id = uuid4()
        self.robots[follower_id] = _StoredRobot(id=follower_id, type=robot_type, payload=FakePayload(http_url=http_url))
        return follower_id

    async def start(self, http_url: str) -> None:
        await self.store.set_http_url(self.project_id, self.follower, http_url)

    async def session(self, leader_id: UUID, follower_id: UUID | None = None) -> tuple[Any, Any]:
        """Start a session of the follower and the leader the way the runtime websocket does."""
        follower = self.robots[follower_id or self.follower]
        return await self.manager.point_leader_at_follower(self.project_id, follower, self.robots[leader_id])

    def port(self, leader_id: UUID) -> int:
        return self.robots[leader_id].payload.http_port


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> LeaderWorld:
    return LeaderWorld(monkeypatch)


async def test_a_start_moves_the_leader_that_reads_from_the_simulation(world: LeaderWorld) -> None:
    leader = world.leader(8080)
    world.environment(world.follower, teleoperated_by=leader)

    await world.start("http://127.0.0.1:50001")

    assert world.port(leader) == 50001
    assert world.robots[world.follower].payload.http_url == "http://127.0.0.1:50001"


async def test_a_start_moves_the_followers_teleoperator_from_any_port(world: LeaderWorld) -> None:
    leader = world.leader(9999)
    world.environment(world.follower, teleoperated_by=leader)

    await world.start("http://127.0.0.1:50001")

    assert world.port(leader) == 50001


async def test_another_leader_near_the_follower_on_another_port_stays(world: LeaderWorld) -> None:
    on_previous, elsewhere = world.leader(8080), world.leader(9999)
    world.environment(world.follower, on_previous, elsewhere)

    await world.start("http://127.0.0.1:50001")

    assert [world.port(on_previous), world.port(elsewhere)] == [50001, 9999]


async def test_leaders_away_from_the_follower_or_not_leaders_stay(world: LeaderWorld) -> None:
    elsewhere, other_project, follower_like = (
        world.leader(8080),
        world.leader(8080),
        world.leader(8080, robot_type="Real_Follower"),
    )
    world.environment(world.follower, follower_like)
    world.environment(uuid4(), teleoperated_by=elsewhere)
    world.environment(world.follower, teleoperated_by=other_project, project_id=world.other_project_id)

    await world.start("http://127.0.0.1:50001")

    assert [world.port(elsewhere), world.port(other_project), world.port(follower_like)] == [8080, 8080, 8080]


async def test_a_restart_moves_the_leader_again(world: LeaderWorld) -> None:
    leader = world.leader(8080)
    world.environment(world.follower, leader)

    await world.start("http://127.0.0.1:50001")
    await world.start("http://127.0.0.1:50002")

    assert world.port(leader) == 50002


async def test_a_session_points_a_leader_added_after_the_start_at_the_simulation(world: LeaderWorld) -> None:
    await world.start("http://127.0.0.1:50488")
    leader = world.leader(8080)
    world.environment(world.follower, teleoperated_by=leader)

    follower, session_leader = await world.session(leader)

    assert session_leader.payload.http_port == 50488
    assert follower.payload.http_url == "http://127.0.0.1:50488"
    assert world.port(leader) == 50488


async def test_a_session_on_the_simulations_port_stores_nothing(world: LeaderWorld) -> None:
    leader = world.leader(8080)

    _, session_leader = await world.session(leader)

    assert session_leader.payload.http_port == 8080
    assert world.updates == []


async def test_a_session_waits_for_an_address_write_and_reads_both_robots_again(world: LeaderWorld) -> None:
    leader = world.leader(8080)
    stale_follower, stale_leader = world.robots[world.follower], world.robots[leader]

    async with world.manager.holding_addresses():
        session = asyncio.create_task(
            world.manager.point_leader_at_follower(world.project_id, stale_follower, stale_leader)
        )
        await asyncio.sleep(0.05)
        assert not session.done()
        await world.start("http://127.0.0.1:50001")  # a simulation's start, while it holds the lock

    follower, session_leader = await session

    assert follower.payload.http_url == "http://127.0.0.1:50001"
    assert session_leader.payload.http_port == world.port(leader) == 50001


async def test_a_leader_shared_by_two_environments_reads_from_each_sessions_follower(world: LeaderWorld) -> None:
    await world.start("http://127.0.0.1:50488")
    other = world.simulated_follower("http://127.0.0.1:50500")
    leader = world.leader(8080)
    world.environment(world.follower, teleoperated_by=leader)
    world.environment(other, teleoperated_by=leader)

    ports = []
    for follower in (world.follower, other, world.follower, other):
        _, session_leader = await world.session(leader, follower)
        ports.append((session_leader.payload.http_port, world.port(leader)))
    await world.start("http://127.0.0.1:50600")  # moves it off the other follower's port
    moved = world.port(leader)
    _, session_leader = await world.session(leader, other)

    assert ports == [(50488, 50488), (50500, 50500), (50488, 50488), (50500, 50500)]
    assert moved == 50600
    assert session_leader.payload.http_port == world.port(leader) == 50500
    assert world.robots[other].payload.http_url == "http://127.0.0.1:50500"


@pytest.mark.parametrize("pair", ["hardware-leader", "hardware-follower"])
async def test_a_session_leaves_hardware_alone(world: LeaderWorld, pair: str) -> None:
    if pair == "hardware-leader":
        leader = uuid4()
        world.robots[leader] = _HardwareLeader(id=leader)
        follower = world.follower
    else:
        leader = world.leader(8080)
        follower = world.simulated_follower("http://127.0.0.1:50488", robot_type="Real_Follower")
    given = (world.robots[follower], world.robots[leader])

    assert await world.session(leader, follower) == given
    assert world.updates == []


@pytest.mark.parametrize(
    "http_url", ["http://192.168.1.20:50488", "http://sim.example:50488", "http://127.0.0.1", "not a url:x"]
)
async def test_a_session_of_a_follower_not_on_a_local_port_leaves_the_leader_alone(
    world: LeaderWorld, http_url: str
) -> None:
    follower = world.simulated_follower(http_url)
    leader = world.leader(8080)

    _, session_leader = await world.session(leader, follower)

    assert session_leader.payload.http_port == world.port(leader) == 8080
    assert world.updates == []


async def test_a_failed_leader_store_starts_the_session_with_the_robots_as_given(
    world: LeaderWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    await world.start("http://127.0.0.1:50488")
    leader = world.leader(8080)

    async def fail(*_args: Any) -> None:
        raise RuntimeError("database is gone")

    monkeypatch.setattr(world.store, "point_leader", fail)

    assert await world.session(leader) == (world.robots[world.follower], world.robots[leader])


async def test_a_session_stores_the_simulations_port_on_a_virtual_leader_in_the_database(tmp_path: Path) -> None:
    """The reported case: a virtual leader added after the follower's simulation started keeps 8080."""
    import services.simulation_manager as module

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def session_ctx():
        async with sessions() as session:
            yield session

    registry = RobotCatalogRegistry()
    project_id, follower_id, leader_id = uuid4(), uuid4(), uuid4()
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add(ProjectDB(id=str(project_id), name="Project"))
            await session.commit()
            robots = RobotService(session, registry)
            for robot_id, robot_type, payload in (
                (follower_id, "MuJoCo_SO101_Follower", {"http_url": "http://127.0.0.1:50488"}),
                (leader_id, "MuJoCo_Virtual_Leader", {"http_port": 8080}),
            ):
                await robots.create_robot(
                    project_id,
                    RobotAdapter.validate_python(
                        {"id": str(robot_id), "name": robot_type, "type": robot_type, "payload": payload}
                    ),
                )
            follower = await robots.get_robot_by_id(project_id, follower_id)
            leader = await robots.get_robot_by_id(project_id, leader_id)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(module, "get_async_db_session_ctx", session_ctx)
            manager = SimulationManager(registry)
            _, session_leader = await manager.point_leader_at_follower(project_id, follower, leader)

        async with sessions() as session:
            stored = await RobotService(session, registry).get_robot_by_id(project_id, leader_id)
        assert session_leader.payload.http_port == stored.payload.http_port == 50488
    finally:
        await engine.dispose()


async def test_a_failed_address_write_keeps_nothing_and_the_next_start_moves_every_leader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import services.simulation_manager as module

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'studio.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def session_ctx():
        async with sessions() as session:
            yield session

    registry = RobotCatalogRegistry()
    project_id, environment_id = uuid4(), uuid4()
    follower, first, second = uuid4(), uuid4(), uuid4()

    def robot(robot_id: UUID, robot_type: str, payload: dict[str, Any]) -> Any:
        return RobotAdapter.validate_python(
            {"id": str(robot_id), "name": robot_type, "type": robot_type, "payload": payload}
        )

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with sessions() as session:
            session.add(ProjectDB(id=str(project_id), name="Project"))
            await session.commit()
            robots = RobotService(session, registry)
            await robots.create_robot(
                project_id, robot(follower, "MuJoCo_SO101_Follower", {"http_url": "http://127.0.0.1:50001"})
            )
            for leader in (first, second):
                await robots.create_robot(project_id, robot(leader, "MuJoCo_Virtual_Leader", {"http_port": 50001}))
            session.add(ProjectEnvironmentDB(id=str(environment_id), project_id=str(project_id), name="Rig"))
            session.add(
                EnvironmentRobotDB(
                    environment_id=str(environment_id),
                    robot_id=str(follower),
                    tele_operator_type="robot",
                    tele_operator_robot_id=str(first),
                )
            )
            session.add(EnvironmentRobotDB(environment_id=str(environment_id), robot_id=str(second)))
            await session.commit()

        monkeypatch.setattr(module, "get_async_db_session_ctx", session_ctx)
        update_robot = RobotService.update_robot
        leader_updates = 0

        async def fail_on_the_second_leader(self: RobotService, project_id: UUID, updated: Any, **kwargs: Any) -> Any:
            nonlocal leader_updates
            if updated.id != follower:
                leader_updates += 1
                if leader_updates == 2:
                    raise RuntimeError("database is gone")
            return await update_robot(self, project_id, updated, **kwargs)

        store = module.DatabaseSimulationRobots(registry)
        monkeypatch.setattr(module.RobotService, "update_robot", fail_on_the_second_leader)
        with pytest.raises(RuntimeError):
            await store.set_http_url(project_id, follower, "http://127.0.0.1:50002")
        monkeypatch.setattr(module.RobotService, "update_robot", update_robot)

        async with sessions() as session:
            stored = {r.id: r.payload for r in await RobotService(session, registry).get_robot_list(project_id)}
        assert [stored[follower].http_url, stored[first].http_port, stored[second].http_port] == [
            "http://127.0.0.1:50001",
            50001,
            50001,
        ]

        await store.set_http_url(project_id, follower, "http://127.0.0.1:50003")

        async with sessions() as session:
            stored = {r.id: r.payload for r in await RobotService(session, registry).get_robot_list(project_id)}
        assert [stored[follower].http_url, stored[first].http_port, stored[second].http_port] == [
            "http://127.0.0.1:50003",
            50003,
            50003,
        ]
    finally:
        await engine.dispose()

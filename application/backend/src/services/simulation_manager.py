# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Start, supervise and stop the one simulation Studio runs for a simulated robot.

The robot type's catalog definition describes how to start its simulation
(``definition.simulation``, a ``physicalai_studio_plugin.SimulationLaunch``).
The manager runs that command line as a child process and follows the launch
contract: stdin is a pipe that is never written to, so the simulation stops
when Studio closes it or exits (also on a crash); stdout carries one JSON event
per line; any exit without a ``ready`` event is a failure.

The manager lives in the API process, not in a session worker, so a
simulation outlives the sessions that use it.
"""

import asyncio
import contextlib
import ipaddress
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlsplit
from uuid import UUID

from loguru import logger
from pydantic import BaseModel

from db import get_async_db_session_ctx
from exceptions import (
    SimulationBusyError,
    SimulationInvalidRequestError,
    SimulationNameInUseError,
    SimulationNotSupportedError,
)
from robots.catalog.registry import RobotCatalogRegistry
from robots.robot_service import RobotService
from schemas.project_camera import IPCamera
from schemas.robot import UnavailableRobot
from schemas.simulation import (
    Simulation,
    SimulationDownload,
    SimulationError,
    SimulationPhase,
    SimulationStatus,
    StartSimulationRequest,
    simulation_proxy_path,
)
from services.environment_service import EnvironmentService
from services.project_camera_service import ProjectCameraService

START_TIMEOUT_S = 180.0
"""A starting simulation that prints no event for this long is stopped as failed."""
STOP_EOF_TIMEOUT_S = 10.0
"""How long a simulation gets to exit after its stdin closes before it is terminated."""
STOP_TERM_TIMEOUT_S = 5.0
"""How long a terminated simulation gets to exit before it is killed."""
STORE_ADDRESS_TIMEOUT_S = 10.0
"""How long writing the simulation's address onto its robot may take."""
DRAIN_TIMEOUT_S = 1.0
"""How long to keep reading output after the process exited. A process the simulation spawned can
hold its output open (the runtime's owner inherits stderr), so reading never waits for end of file."""
EXIT_POLL_S = 0.05
LOG_LINES = 500
_LINE_LIMIT = 1 << 20

_ACTIVE = frozenset({SimulationStatus.STARTING, SimulationStatus.RUNNING, SimulationStatus.STOPPING})


class SimulationRobots(Protocol):
    """Where the manager reads the robot to simulate and writes back its address."""

    async def get(self, project_id: UUID, robot_id: UUID) -> Any:
        """Return the project's robot, or raise ``ResourceNotFoundError``."""
        ...

    async def set_http_url(self, project_id: UUID, robot_id: UUID, http_url: str) -> None:
        """Store *http_url* on the robot's payload."""
        ...

    async def point_leader(self, project_id: UUID, follower_id: UUID, leader_id: UUID) -> tuple[Any, Any] | None:
        """Store the port of the follower's simulation on a leader that reads from a simulation.

        Returns the follower and the leader as stored, or ``None`` if the follower is not simulated
        on this machine or the leader does not read from a simulation.
        """
        ...


class DatabaseSimulationRobots:
    """Read and update robots through ``RobotService`` with a session of their own."""

    def __init__(self, registry: RobotCatalogRegistry) -> None:
        self._registry = registry

    async def get(self, project_id: UUID, robot_id: UUID) -> Any:
        async with get_async_db_session_ctx() as session:
            return await RobotService(session, self._registry).get_robot_by_id(project_id, robot_id)

    async def set_http_url(self, project_id: UUID, robot_id: UUID, http_url: str) -> None:
        async with get_async_db_session_ctx() as session:
            service = RobotService(session, self._registry)
            robot = await service.get_robot_by_id(project_id, robot_id)
            if isinstance(robot, UnavailableRobot) or "http_url" not in type(robot.payload).model_fields:
                return
            # One transaction for the cameras, the leaders and the robot: the leaders are matched on
            # the robot's previous address, so they must not move without it. If any of it fails,
            # none of it is kept, and the next start moves them all from the address still stored.
            cameras = ProjectCameraService(session, self._registry)
            await _move_simulation_cameras(cameras, project_id, robot_id, http_url, commit=False)
            if robot.payload.http_url != http_url:
                environments = EnvironmentService(session, self._registry)
                await _move_simulation_leaders(
                    service, environments, self._registry, project_id, robot_id, robot.payload.http_url, http_url
                )
                payload = robot.payload.model_copy(update={"http_url": http_url})
                await service.update_robot(project_id, robot.model_copy(update={"payload": payload}), commit=False)
            await session.commit()

    async def point_leader(self, project_id: UUID, follower_id: UUID, leader_id: UUID) -> tuple[Any, Any] | None:
        async with get_async_db_session_ctx() as session:
            service = RobotService(session, self._registry)
            follower = await service.get_robot_by_id(project_id, follower_id)
            leader = await service.get_robot_by_id(project_id, leader_id)
            port, new_port = _virtual_leader_port(self._registry, leader), _simulation_port(self._registry, follower)
            if port is None or new_port is None:
                return None
            if port != new_port:
                payload = leader.payload.model_copy(update={"http_port": new_port})
                leader = leader.model_copy(update={"payload": payload})
                await service.update_robot(project_id, leader)
                logger.info(
                    "Pointed leader {!r} at the simulation of {!r}: port {}, was {}",
                    leader.name,
                    follower.name,
                    new_port,
                    port,
                )
            return follower, leader


async def _move_simulation_cameras(
    cameras: ProjectCameraService, project_id: UUID, robot_id: UUID, new_url: str, *, commit: bool = True
) -> None:
    """Point the IP cameras added from a robot's simulation at the simulation's new address.

    Simulations start on free ports, so the address changes on every start. A camera added from a
    simulation records its robot, and only that robot's cameras move, from whatever address they
    were on (the robot's stored address may have changed since). A camera without a robot never
    moves: its address cannot tell which simulation it came from.
    """
    for camera in await cameras.get_camera_list(project_id):
        if not isinstance(camera, IPCamera) or camera.payload.simulation_robot_id != robot_id:
            continue
        base, separator, stream = camera.payload.url.partition("/cameras/")
        if not separator or base == new_url:
            continue
        url = f"{new_url}/cameras/{stream}"
        payload = camera.payload.model_copy(update={"url": url})
        await cameras.update_camera(
            project_id, camera.model_copy(update={"payload": payload, "fingerprint": {"url": url}}), commit=commit
        )


async def _move_simulation_leaders(
    robots: RobotService,
    environments: EnvironmentService,
    registry: RobotCatalogRegistry,
    project_id: UUID,
    robot_id: UUID,
    old_url: str,
    new_url: str,
) -> None:
    """Point the leaders that read from a robot's simulation at the simulation's new port.

    A leader of any robot type (role ``leader``) with an integer ``http_port`` payload field reads
    from a simulation on this machine. The robot's teleoperator in an environment reads from this
    simulation and moves with it, from any port: a session of the robot points it here anyway (see
    ``SimulationManager.point_leader_at_follower``), so the form shows the port it will read from.
    Another leader in an environment with the robot moves only from the port of the robot's
    previous address; one on another port reads from another simulation and stays. Only flushed:
    the caller commits.
    """
    old_port, new_port = _local_port(old_url), _local_port(new_url)
    if new_port is None or old_port == new_port:
        return
    nearby: set[UUID] = set()
    teleoperators: set[UUID] = set()
    for environment in await environments.get_environment_list(project_id):
        members = {
            member
            for configuration in environment.robots
            for member in (configuration.robot_id, getattr(configuration.tele_operator, "robot_id", None))
            if member is not None
        }
        if robot_id in members:
            nearby |= members - {robot_id}
        teleoperators |= {
            teleoperator
            for configuration in environment.robots
            if configuration.robot_id == robot_id
            and (teleoperator := getattr(configuration.tele_operator, "robot_id", None)) is not None
        }
    for leader_id in nearby:
        leader = await robots.get_robot_by_id(project_id, leader_id)
        port = _virtual_leader_port(registry, leader)
        if port is None or port == new_port or (leader_id not in teleoperators and port != old_port):
            continue
        payload = leader.payload.model_copy(update={"http_port": new_port})
        await robots.update_robot(project_id, leader.model_copy(update={"payload": payload}), commit=False)
        logger.info("Moved leader {!r} from port {} to its simulation's port {}", leader.name, port, new_port)


def _virtual_leader_port(registry: RobotCatalogRegistry, robot: Any) -> int | None:
    """Return the port a leader reads from a simulation on, or ``None`` for any other robot.

    A leader of any robot type (role ``leader``) with an integer ``http_port`` payload field reads
    from a simulation on this machine; a hardware leader has no such field.
    """
    if isinstance(robot, UnavailableRobot) or getattr(registry.get_definition(robot.type), "role", None) != "leader":
        return None
    port = getattr(robot.payload, "http_port", None)
    return port if type(port) is int else None


def _simulation_port(registry: RobotCatalogRegistry, robot: Any) -> int | None:
    """Return the port of a simulated robot's simulation on this machine, or ``None``.

    ``None`` for a robot type without a simulation, and for an address on another host or without a
    port. The stored address is the simulation's: Studio stores it once a simulation it started is
    ready, and one started outside Studio runs on the address the robot is configured with.
    """
    if isinstance(robot, UnavailableRobot) or getattr(registry.get_definition(robot.type), "simulation", None) is None:
        return None
    http_url = getattr(robot.payload, "http_url", None)
    return _local_port(http_url) if isinstance(http_url, str) else None


def _local_port(http_url: str) -> int | None:
    """Return the port of an address on this machine, or ``None`` for another host or no port."""
    try:
        url = urlsplit(http_url)
        host, port = url.hostname or "", url.port
    except ValueError:
        return None
    if port is None:
        return None
    if host == "localhost":
        return port
    try:
        return port if ipaddress.ip_address(host).is_loopback else None
    except ValueError:
        return None


@dataclass
class _Child:
    process: asyncio.subprocess.Process
    owner_name: str
    stop_requested: bool = False
    deadline: float = 0.0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: list[asyncio.Task] = field(default_factory=list)


class SimulationManager:
    """Owns the one simulation Studio runs at a time; see the module docstring."""

    def __init__(
        self,
        registry: RobotCatalogRegistry,
        robots: SimulationRobots | None = None,
        *,
        start_timeout_s: float = START_TIMEOUT_S,
        stop_eof_timeout_s: float = STOP_EOF_TIMEOUT_S,
        stop_term_timeout_s: float = STOP_TERM_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._robots = robots or DatabaseSimulationRobots(registry)
        self._start_timeout_s = start_timeout_s
        self._stop_eof_timeout_s = stop_eof_timeout_s
        self._stop_term_timeout_s = stop_term_timeout_s
        self._lock = asyncio.Lock()
        self._state = Simulation()
        self._child: _Child | None = None
        self._logs: deque[str] = deque(maxlen=LOG_LINES)
        self._subscribers: set[asyncio.Queue[Simulation]] = set()
        self._background: set[asyncio.Task] = set()
        self._address_writes: set[asyncio.Task] = set()
        self._address_lock = asyncio.Lock()

    # -- reading -----------------------------------------------------------

    def state(self) -> Simulation:
        """Return a snapshot of the simulation's state."""
        return self._state.model_copy(deep=True)

    def logs(self) -> list[str]:
        """Return the latest log lines of the current or last simulation."""
        return list(self._logs)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Simulation]]:
        """Yield a queue that holds the latest state: the current one first, then every change.

        A slow reader only misses intermediate states, never the latest one.
        """
        queue: asyncio.Queue[Simulation] = asyncio.Queue(maxsize=1)
        queue.put_nowait(self.state())
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    @contextlib.asynccontextmanager
    async def updating_robot(self, project_id: UUID, robot_id: UUID) -> AsyncIterator[bool]:
        """Hold off storing a simulation's address while a robot is updated.

        Yields whether Studio is starting, running or stopping this robot's simulation. Its address
        is Studio's then: whoever updates the robot keeps the stored one.
        """
        async with self.holding_addresses() as simulated_by_studio:
            yield simulated_by_studio(project_id, robot_id)

    @contextlib.asynccontextmanager
    async def holding_addresses(self) -> AsyncIterator[Callable[[UUID, UUID], bool]]:
        """Hold off storing simulation addresses, and so moving and adopting cameras.

        One lock, never taken while it is held, for robot and camera updates, address writes and
        additions of a simulation's cameras alike. Yields whether Studio is starting, running or
        stopping the simulation of a project's robot.
        """
        async with self._address_lock:
            yield self._simulated_by_studio

    async def point_leader_at_follower(self, project_id: UUID, follower: Any, leader: Any) -> tuple[Any, Any]:
        """Point a leader that reads from a simulation at the follower's, for a session of the two.

        Returns the follower and the leader to start the session with. If the follower is simulated
        on this machine (its type has a simulation and its address is on this machine, with a port)
        and the leader reads from a simulation (see ``_virtual_leader_port``), the leader reads from
        the port of the follower's address, stored on it if it changed, and both are read again
        under the address lock, so that a simulation's start cannot move either in between. A leader
        added, linked or edited after the simulation started, or shared with an environment of
        another follower, is so pointed at this session's follower; a session of the other follower
        points it back. Otherwise, or if storing fails, both are returned as given.
        """
        if _virtual_leader_port(self._registry, leader) is None or (
            getattr(self._registry.get_definition(follower.type), "simulation", None) is None
        ):
            return follower, leader
        try:
            async with self._address_lock:
                stored = await asyncio.wait_for(
                    self._robots.point_leader(project_id, follower.id, leader.id), STORE_ADDRESS_TIMEOUT_S
                )
        except Exception:
            logger.exception("Could not point leader {!r} at the simulation of {!r}", leader.name, follower.name)
            return follower, leader
        return stored if stored is not None else (follower, leader)

    def _simulated_by_studio(self, project_id: UUID, robot_id: UUID) -> bool:
        state = self._state
        return state.status in _ACTIVE and state.project_id == project_id and state.robot_id == robot_id

    # -- commands ----------------------------------------------------------

    async def start(self, request: StartSimulationRequest) -> Simulation:
        """Start the simulation of a project's robot; returns the ``starting`` state.

        Raises:
            ResourceNotFoundError: The robot does not exist.
            SimulationNotSupportedError: The robot type has no simulation, or the robot points
                at another host.
            SimulationInvalidRequestError: The scene or seed is not valid for the robot type.
            SimulationBusyError: Studio runs a simulation and ``replace`` is false.
            SimulationNameInUseError: A simulation Studio did not start runs under the robot's name.
        """
        async with self._lock:
            robot = await self._robots.get(request.project_id, request.robot_id)
            if isinstance(robot, UnavailableRobot):
                raise SimulationNotSupportedError(
                    f"Robot {robot.name!r} cannot be simulated: the plugin for {robot.type!r} is not installed."
                )
            definition = self._registry.get_definition(robot.type)
            launch = getattr(definition, "simulation", None)
            if definition is None or launch is None:
                raise SimulationNotSupportedError(f"Robot {robot.name!r} ({robot.type}) has no simulation.")
            _check_local(robot.name, robot.payload)
            try:
                argv = launch.argv(request.scene, robot.payload, request.seed)
                owner_name = launch.owner_name(robot.payload)
            except ValueError as error:
                raise SimulationInvalidRequestError(str(error)) from error

            if self._state.status in _ACTIVE:
                if not request.replace:
                    raise SimulationBusyError(
                        project_id=str(self._state.project_id),
                        robot_id=str(self._state.robot_id),
                        scene=str(self._state.scene),
                    )
                await self._stop_and_wait()
            elif self._child is not None:
                # A failed simulation that is still exiting.
                await self._stop_and_wait()

            if await _is_online(definition, robot.payload):
                raise SimulationNameInUseError(owner_name)

            # A stopped simulation's address may still be being written. Let it land first, so that
            # it cannot overwrite the address of the one starting now.
            await self._finish_address_writes()

            await self._spawn(argv, owner_name, request, robot.type)
            return self.state()

    async def stop(self) -> Simulation:
        """Start stopping the simulation, if any, and clear a failed state; returns the new state.

        Stopping finishes in the background: the state goes to ``idle`` once the process exited.
        """
        async with self._lock:
            child = self._child
            if child is None:
                if self._state.status is not SimulationStatus.IDLE:
                    self._set(Simulation())
                return self.state()
            self._request_stop(child)
            return self.state()

    async def shutdown(self) -> None:
        """Stop the simulation, if any, and wait until it exited. Called when Studio shuts down."""
        async with self._lock:
            await self._stop_and_wait()
        for task in list(self._background):
            task.cancel()

    # -- process -----------------------------------------------------------

    async def _spawn(self, argv: list[str], owner_name: str, request: StartSimulationRequest, robot_type: str) -> None:
        self._logs.clear()
        logger.info("Starting simulation {!r}: {}", owner_name, argv)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_LINE_LIMIT,
            )
        except OSError as error:
            raise SimulationNotSupportedError(f"Could not start the simulation: {error}") from error

        child = _Child(process=process, owner_name=owner_name, deadline=time.monotonic() + self._start_timeout_s)
        self._child = child
        self._set(
            Simulation(
                status=SimulationStatus.STARTING,
                project_id=request.project_id,
                robot_id=request.robot_id,
                robot_type=robot_type,
                owner_name=owner_name,
                scene=request.scene,
                seed=request.seed,
                started_at=datetime.now(UTC),
            )
        )
        stdout = asyncio.create_task(self._read_events(child))
        stderr = asyncio.create_task(self._read_log(child))
        child.tasks = [
            stdout,
            stderr,
            asyncio.create_task(self._watch_start(child)),
            asyncio.create_task(self._wait_exit(child, stdout, stderr)),
        ]

    async def _read_events(self, child: _Child) -> None:
        if child.process.stdout is None:
            return
        stdout = child.process.stdout
        while True:
            try:
                raw = await stdout.readline()
            except ValueError:
                # Longer than the line limit: readline dropped it; keep reading so the pipe never fills.
                self._log(f"(dropped an output line longer than {_LINE_LIMIT} bytes)")
                continue
            if not raw:
                return
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if not isinstance(event, dict) or "event" not in event:
                self._log(line)
                continue
            if child is self._child:
                child.deadline = time.monotonic() + self._start_timeout_s
                try:
                    self._on_event(child, event)
                except Exception:
                    logger.exception("Could not handle simulation event {}", line)

    async def _read_log(self, child: _Child) -> None:
        if child.process.stderr is None:
            return
        stderr = child.process.stderr
        while True:
            try:
                raw = await stderr.readline()
            except ValueError:
                self._log(f"(dropped a log line longer than {_LINE_LIMIT} bytes)")
                continue
            if not raw:
                return
            self._log(raw.decode(errors="replace").rstrip())

    def _log(self, line: str) -> None:
        self._logs.append(line)
        logger.debug("simulation: {}", line)

    def _on_event(self, child: _Child, event: dict[str, Any]) -> None:
        if child.stop_requested or self._state.status not in {SimulationStatus.STARTING, SimulationStatus.RUNNING}:
            return
        kind = event.get("event")
        if kind == "phase" and self._state.status is SimulationStatus.STARTING:
            self._on_phase(event)
        elif kind == "ready" and self._state.status is SimulationStatus.STARTING:
            self._on_ready(child, event)
        elif kind == "error":
            self._fail(child, str(event.get("message") or "The simulation reported an error."))

    def _on_phase(self, event: dict[str, Any]) -> None:
        try:
            phase = SimulationPhase(event.get("phase"))
        except ValueError:
            return
        download = None
        if phase is SimulationPhase.FETCH and isinstance(event.get("bytes"), int):
            total = event.get("total")
            download = SimulationDownload(bytes=event["bytes"], total=total if isinstance(total, int) else None)
        self._update(phase=phase, download=download)

    def _on_ready(self, child: _Child, event: dict[str, Any]) -> None:
        name = event.get("name")
        if name != child.owner_name:
            self._fail(child, f"The simulation started as {name!r}, not as {child.owner_name!r}.")
            return
        http_url = event.get("http_url")
        cameras = event.get("cameras")
        pid = event.get("pid")
        running = {
            "status": SimulationStatus.RUNNING,
            "phase": None,
            "download": None,
            "pid": pid if isinstance(pid, int) else child.process.pid,
            "http_url": http_url if isinstance(http_url, str) else None,
            "viewer_url": event.get("viewer_url") if isinstance(event.get("viewer_url"), str) else None,
            "viewer_proxy_url": (
                f"{simulation_proxy_path(self._state.project_id, self._state.robot_id)}/viewer/"
                if isinstance(event.get("viewer_url"), str) and self._state.project_id and self._state.robot_id
                else None
            ),
            "cameras": [str(camera) for camera in cameras] if isinstance(cameras, list) else [],
        }
        task = self._spawn_background(self._finish_ready(child, running))
        self._address_writes.add(task)
        task.add_done_callback(self._address_writes.discard)

    async def _finish_ready(self, child: _Child, running: dict[str, Any]) -> None:
        """Store the simulation's address, then report it running.

        In this order, so that whoever reacts to ``running`` (the UI refetching the robot and its
        cameras) already sees the new address.
        """
        http_url = running["http_url"]
        if http_url is not None and self._state.project_id and self._state.robot_id:
            await self._store_address(self._state.project_id, self._state.robot_id, http_url)
        if child is self._child and not child.stop_requested and self._state.status is SimulationStatus.STARTING:
            self._update(**running)

    async def _finish_address_writes(self) -> None:
        """Wait until the addresses being stored are stored (each write is bounded by its own timeout)."""
        if self._address_writes:
            await asyncio.gather(*self._address_writes, return_exceptions=True)

    async def _store_address(self, project_id: UUID, robot_id: UUID, http_url: str) -> None:
        try:
            async with self._address_lock:
                await asyncio.wait_for(
                    self._robots.set_http_url(project_id, robot_id, http_url), STORE_ADDRESS_TIMEOUT_S
                )
        except Exception:
            logger.exception("Could not store the simulation's address {} on its robot", http_url)

    def _fail(self, child: _Child, message: str) -> None:
        """Mark the simulation failed and stop its process; the exit keeps the failed state."""
        logger.warning("Simulation {!r} failed: {}", child.owner_name, message)
        self._update(
            status=SimulationStatus.FAILED,
            error=SimulationError(message=message, phase=self._state.phase),
            download=None,
        )
        self._terminate(child)

    async def _watch_start(self, child: _Child) -> None:
        while self._state.status is SimulationStatus.STARTING and child is self._child and not child.stop_requested:
            remaining = child.deadline - time.monotonic()
            if remaining <= 0:
                phase = f" in the {self._state.phase} phase" if self._state.phase else ""
                self._fail(child, f"The simulation made no progress{phase} for {self._start_timeout_s:.0f} s.")
                return
            await asyncio.sleep(remaining)

    async def _wait_exit(self, child: _Child, *readers: asyncio.Task) -> None:
        exit_code = await _exit_code(child.process)
        _, pending = await asyncio.wait(readers, timeout=DRAIN_TIMEOUT_S)
        for task in child.tasks:
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
        for task in readers:
            if task not in pending and not task.cancelled() and task.exception() is not None:
                logger.opt(exception=task.exception()).error("Reading the simulation's output failed")
        logger.info("Simulation {!r} exited with code {}", child.owner_name, exit_code)
        if child is self._child:
            self._child = None
            self._on_exit(child, exit_code)
        child.done.set()

    def _on_exit(self, child: _Child, exit_code: int) -> None:
        status = self._state.status
        if child.stop_requested:
            self._set(Simulation())
            return
        if status is SimulationStatus.FAILED:
            if self._state.error is not None:
                self._update(error=self._state.error.model_copy(update={"exit_code": exit_code}))
            return
        if status is SimulationStatus.RUNNING and exit_code == 0:
            self._set(Simulation())
            return
        last_line = next((line for line in reversed(self._logs) if line.strip()), None)
        if status is SimulationStatus.STARTING:
            message = f"The simulation exited with code {exit_code} before it was ready."
        else:
            message = f"The simulation stopped unexpectedly with code {exit_code}."
        if last_line:
            message = f"{message} {last_line}"
        self._update(
            status=SimulationStatus.FAILED,
            error=SimulationError(message=message, phase=self._state.phase, exit_code=exit_code),
            download=None,
        )

    def _request_stop(self, child: _Child) -> None:
        """Close the child's stdin and escalate in the background if it does not exit."""
        if child.stop_requested:
            return
        child.stop_requested = True
        self._update(status=SimulationStatus.STOPPING, phase=None, download=None, error=None)
        self._close_stdin(child)
        self._spawn_background(self._escalate(child))

    def _terminate(self, child: _Child) -> None:
        """Stop a child without marking the stop as requested (its failure stays visible)."""
        self._close_stdin(child)
        self._spawn_background(self._escalate(child))

    @staticmethod
    def _close_stdin(child: _Child) -> None:
        stdin = child.process.stdin
        if stdin is not None and not stdin.is_closing():
            stdin.close()

    async def _escalate(self, child: _Child) -> None:
        if await self._exited(child, self._stop_eof_timeout_s) or child.process.returncode is not None:
            return
        logger.warning("Simulation {!r} ignored end of input; terminating it", child.owner_name)
        with contextlib.suppress(ProcessLookupError):
            child.process.terminate()
        if await self._exited(child, self._stop_term_timeout_s) or child.process.returncode is not None:
            return
        logger.warning("Simulation {!r} ignored SIGTERM; killing it", child.owner_name)
        with contextlib.suppress(ProcessLookupError):
            child.process.kill()

    @staticmethod
    async def _exited(child: _Child, timeout_s: float) -> bool:
        try:
            await asyncio.wait_for(child.done.wait(), timeout_s)
        except TimeoutError:
            return False
        return True

    async def _stop_and_wait(self) -> None:
        child = self._child
        if child is None:
            return
        self._request_stop(child)
        budget = self._stop_eof_timeout_s + self._stop_term_timeout_s + DRAIN_TIMEOUT_S + 5.0
        if not await self._exited(child, budget):
            # SIGKILL was sent, so this only happens if the OS never reports the exit.
            logger.error("Simulation {!r} (pid {}) did not exit", child.owner_name, child.process.pid)
            if child is self._child:
                self._child = None
                self._set(Simulation())

    def _spawn_background(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # -- state -------------------------------------------------------------

    def _update(self, **changes: Any) -> None:
        self._set(self._state.model_copy(update=changes))

    def _set(self, state: Simulation) -> None:
        self._state = state
        for queue in self._subscribers:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(state.model_copy(deep=True))


async def _exit_code(process: asyncio.subprocess.Process) -> int:
    """Wait until *process* exited and return its exit code.

    Not ``process.wait()``: asyncio only returns from it once every pipe is closed, and a process
    the simulation spawned can keep its stderr open long after the simulation exited.
    """
    while True:
        if process.returncode is not None:
            return process.returncode
        await asyncio.sleep(EXIT_POLL_S)


def _check_local(robot_name: str, payload: BaseModel) -> None:
    """Refuse robots whose payload points at another host: Studio starts simulations on this machine."""
    if getattr(payload, "allow_remote", False) is True:
        raise SimulationNotSupportedError(
            f"Robot {robot_name!r} allows remote simulations. Studio only starts simulations on this machine."
        )
    http_url = getattr(payload, "http_url", None)
    if not isinstance(http_url, str):
        return
    host = urlsplit(http_url).hostname or ""
    if host == "localhost":
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise SimulationNotSupportedError(
        f"Robot {robot_name!r} attaches to a simulation on {host}. Studio only starts simulations on this machine."
    )


async def _is_online(definition: Any, payload: BaseModel) -> bool:
    probe = getattr(definition, "probe", None)
    if probe is None:
        return False
    try:
        return bool(await probe.is_online(payload))
    except Exception:
        logger.exception("Could not check whether a simulation already runs; assuming it does not")
        return False

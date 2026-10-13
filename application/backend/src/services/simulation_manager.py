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
from collections.abc import AsyncIterator
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
)
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
            old_url = robot.payload.http_url
            if old_url == http_url:
                return
            # Cameras first: if moving them fails, the robot still has the old address and the next
            # start moves the rest.
            await _move_simulation_cameras(ProjectCameraService(session, self._registry), project_id, old_url, http_url)
            payload = robot.payload.model_copy(update={"http_url": http_url})
            await service.update_robot(project_id, robot.model_copy(update={"payload": payload}))


async def _move_simulation_cameras(cameras: ProjectCameraService, project_id: UUID, old_url: str, new_url: str) -> None:
    """Point the project's IP cameras on a simulation's streams at its new address.

    Simulations start on free ports, so the address changes on every start.
    """
    prefix = f"{old_url}/cameras/"
    for camera in await cameras.get_camera_list(project_id):
        if not isinstance(camera, IPCamera) or not camera.payload.url.startswith(prefix):
            continue
        url = f"{new_url}/cameras/{camera.payload.url.removeprefix(prefix)}"
        await cameras.update_camera(
            project_id,
            camera.model_copy(
                update={"payload": camera.payload.model_copy(update={"url": url}), "fingerprint": {"url": url}}
            ),
        )


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
            "cameras": [str(camera) for camera in cameras] if isinstance(cameras, list) else [],
        }
        self._spawn_background(self._finish_ready(child, running))

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

    async def _store_address(self, project_id: UUID, robot_id: UUID, http_url: str) -> None:
        try:
            await asyncio.wait_for(self._robots.set_http_url(project_id, robot_id, http_url), STORE_ADDRESS_TIMEOUT_S)
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

    def _spawn_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

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

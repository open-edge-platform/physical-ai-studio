"""Loading an environment: describe it, then connect it.

``describe_environment`` works out what to load without touching hardware: the
store spec, the follower and leader recipes and the cameras. The joints come
from the driver, not the database: a robot row stores a type and a port, and
only the driver for that type knows its joints and their order. Building a
driver does not touch the device; only ``connect()`` does.

An :class:`Environment` then holds it: the device locks, the follower's
``RobotWorker``, a ``CameraWorker`` per camera and the leader as a plain
driver, all connected in parallel and released in reverse. Only teleop needs the
leader, so one that fails to connect reports so and leaves the environment running. They write to the
session's store, which outlives them. It has its own connection to the
session's bus, where its devices report as ``environment/<device>`` and it
reports as ``environment``.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import UUID

import anyio
from loguru import logger

from exceptions import RuntimeSessionCommandError
from robots.features import sanitize_camera_name
from runtime_session.bus import Bus, Subscription, publish_status, status_key
from runtime_session.store import FeatureSpec, Store
from schemas.runtime_session import CommandDone, ReloadEnvironment, SessionCamera, SessionEnvironment, StatusMessage
from workers.base import WorkerStartupError
from workers.camera_worker import CameraWorker, camera_image_spec
from workers.robot_worker import DEFAULT_ROBOT_HZ, RobotRecipe, RobotWorker, claim_robot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Mapping
    from contextlib import AbstractAsyncContextManager

    from physicalai.robot import Robot

    from robots.robot_client_factory import RobotClientFactory
    from schemas.environment import EnvironmentWithRelations
    from schemas.project_camera import Camera


UNSAVED_ENVIRONMENT_ID = UUID(int=0)
"""The id an environment from the form carries, which is not saved yet."""


@dataclass(frozen=True, slots=True)
class EnvironmentDescription:
    environment_id: UUID | None
    name: str
    robot_names: Mapping[UUID, str]
    """Display names of the follower and leader, for claim errors."""
    spec: FeatureSpec
    follower: RobotRecipe
    leader: RobotRecipe | None
    """Connected as a plain driver by the session, for controls to borrow."""
    cameras: Mapping[str, Camera]
    """Camera rows by store key, the key a dataset records them under."""

    def session_environment(self) -> SessionEnvironment:
        """What the session's ``state`` message says about this environment."""
        return SessionEnvironment(
            environment_id=self.environment_id,
            name=self.name,
            robot_id=self.follower.robot_id,
            leader_id=self.leader.robot_id if self.leader is not None else None,
            joint_names=list(self.spec.joint_names),
            cameras=[
                SessionCamera(
                    camera_id=camera.id,
                    name=camera.name,
                    key=key,
                    height=self.spec.observations[key].height,
                    width=self.spec.observations[key].width,
                )
                for key, camera in self.cameras.items()
            ],
        )


async def describe_environment(
    environment: EnvironmentWithRelations, factory: RobotClientFactory
) -> EnvironmentDescription:
    """Build the follower's driver (unconnected) and derive the session's spec.

    Raises:
        ValueError: The environment does not have exactly one robot, or two
            cameras share a store key.
    """
    if len(environment.robots) != 1:
        raise ValueError(
            f"Environment {environment.name!r} has {len(environment.robots)} robots; "
            "a session drives exactly one robot for now"
        )
    robot = environment.robots[0].robot
    driver, _definition = await factory.build_robot_driver(robot, factory)
    follower = RobotRecipe.from_driver(robot.id, driver)
    robot_names = {robot.id: robot.name}
    leader = None
    if (leader_robot := getattr(environment.robots[0].tele_operator, "robot", None)) is not None:
        leader_driver, _definition = await factory.build_robot_driver(leader_robot, factory)
        leader = RobotRecipe.from_driver(leader_robot.id, leader_driver)
        robot_names[leader_robot.id] = leader_robot.name

    cameras: dict[str, Camera] = {}
    for camera in environment.cameras:
        key = sanitize_camera_name(camera.name)
        if key in cameras:
            raise ValueError(f"Cameras {cameras[key].name!r} and {camera.name!r} would both be recorded as {key!r}")
        cameras[key] = camera
    spec = FeatureSpec(follower.joint_names, {key: camera_image_spec(camera) for key, camera in cameras.items()})
    return EnvironmentDescription(
        environment_id=None if environment.id == UNSAVED_ENVIRONMENT_ID else environment.id,
        name=environment.name,
        robot_names=robot_names,
        spec=spec,
        follower=follower,
        leader=leader,
        cameras=cameras,
    )


COMPONENT = "environment"
"""What the environment reports as; its devices report as ``environment/<device>``."""


RELOAD = "rpc/environment/reload"
"""Where the environment takes a request to reconnect its devices."""


class Environment:
    """An environment's devices, writing to the session's store.

    ``async with`` joins the bus, answers ``RELOAD`` and tries to connect them,
    until it is left, which releases them. A device that fails, connecting or
    later, reports it on the bus. Nothing is recoverable yet, so the environment
    then reports itself failed and keeps holding what it has until it is
    reloaded or released. A reload connects the devices again in place.
    """

    def __init__(self, description: EnvironmentDescription, session_id: str, store: Store) -> None:
        self.description = description
        self.session_id = session_id
        self.store = store
        self.leader: Robot | None = None
        """Connected, for controls to borrow; ``None`` while the devices are not connected."""
        self._bus: Bus | None = None
        self._devices: AsyncExitStack | None = None

    async def __aenter__(self) -> Self:
        """Join the bus and try to connect the devices; if that fails, it is reported failed, to reload."""
        self._bus = Bus.join(self.session_id)
        self._bus.serve(RELOAD, ReloadEnvironment, self.reload)  # undeclared with the bus
        try:
            await self._connect()
        except Exception:
            logger.exception("Environment {!r} could not connect", self.description.name)
        except BaseException:
            self._bus.close()
            raise
        return self

    async def __aexit__(self, *_exc: object) -> None:
        try:
            await self._disconnect()
        finally:
            publish_status(self.bus, COMPONENT, "stopped")
            self.bus.close()

    @property
    def bus(self) -> Bus:
        """The environment's own connection to the session's bus, while it is entered."""
        if self._bus is None:
            raise RuntimeError("The environment is not entered")
        return self._bus

    async def reload(self, _request: ReloadEnvironment | None = None) -> CommandDone:
        """Disconnect the devices and connect them again. If that fails, stay as failed, to try again.

        Raises:
            RuntimeSessionCommandError: A device failed to connect.
        """
        await self._disconnect()
        try:
            await self._connect()
        except Exception as error:
            raise RuntimeSessionCommandError(_reason(error), error_code="device_connection_failed") from error
        return CommandDone()

    async def _connect(self) -> None:
        """Connect every device; on any failure, release what was acquired, report it and raise.

        Raises:
            RobotDeviceAlreadyOwnedError: Something outside Studio's sessions holds a robot.
            WorkerStartupError: A device failed to connect.
        """
        description, store = self.description, self.store
        publish_status(self.bus, COMPONENT, "starting")
        async with AsyncExitStack() as stack:
            try:
                # Subscribed before any device starts, so no failure is missed; anew each time, so
                # what the devices said before a reload is not heard again.
                devices = stack.enter_context(self.bus.subscribe(status_key(f"{COMPONENT}/*"), StatusMessage))
                stack.enter_context(claim_robot(description.follower))
                if description.leader is not None:
                    stack.enter_context(claim_robot(description.leader))
                # shortcut: a killed robot process reports nothing; await its ``wait_stopped()`` if that happens.
                follower = RobotWorker(
                    description.follower,
                    store.name,
                    store.spec,
                    hz=DEFAULT_ROBOT_HZ,
                    session_id=self.session_id,
                    component=f"{COMPONENT}/follower",
                )
                managers: dict[str, AbstractAsyncContextManager[Any]] = {"follower": follower}
                for key, row in description.cameras.items():
                    managers[key] = CameraWorker(
                        row, store, key, session_id=self.session_id, component=f"{COMPONENT}/{key}"
                    )
                if description.leader is not None:
                    managers["leader"] = _connected_driver(description.leader)
                started = await _enter_concurrently(stack, self.bus, managers, optional=_OPTIONAL)
            except Exception as error:
                publish_status(self.bus, COMPONENT, "failed", _reason(error))
                raise
            handler = asyncio.create_task(self._handle_failures(devices))
            stack.callback(handler.cancel)
            self.leader = started.get("leader")
            self._devices = stack.pop_all()
        publish_status(self.bus, COMPONENT, "running")

    async def _disconnect(self) -> None:
        """Release the devices, if they are connected."""
        devices, self._devices, self.leader = self._devices, None, None
        if devices is not None:
            await devices.aclose()

    async def _handle_failures(self, devices: Subscription[StatusMessage]) -> None:
        """Recover a device that failed, or fail with it. Nothing is recoverable yet."""
        while True:
            status = await devices.next()
            if status.state == "failed":
                device = status.component.removeprefix(f"{COMPONENT}/")
                if device in _OPTIONAL:
                    continue
                logger.error("Environment {!r}: {} failed: {}", self.description.name, device, status.error)
                publish_status(self.bus, COMPONENT, "failed", f"{device}: {status.error}")
                return


_OPTIONAL = frozenset({"leader"})
"""Devices the environment runs without; ``leader`` stays ``None`` until a reload connects it."""


def _reason(error: Exception) -> str:
    if isinstance(error, WorkerStartupError):
        return error.reason
    return getattr(error, "message", None) or str(error) or type(error).__name__


@asynccontextmanager
async def _connected_driver(recipe: RobotRecipe) -> AsyncIterator[Robot]:
    """A plain driver in this process: no worker, no loop, no store slot."""
    from physicalai.config import instantiate

    driver = cast("Robot", await asyncio.to_thread(instantiate, recipe.driver))
    await asyncio.to_thread(driver.connect)
    try:
        yield driver
    finally:
        with anyio.CancelScope(shield=True):
            await asyncio.to_thread(driver.disconnect)


async def _enter_concurrently(
    stack: AsyncExitStack,
    bus: Bus,
    devices: Mapping[str, AbstractAsyncContextManager[Any]],
    optional: Collection[str] = (),
) -> dict[str, Any]:
    """Enter every device at once, reporting each; on a failure, exit those that started and raise the first error.

    An ``optional`` device that fails is reported and left out, as ``None``.
    """
    for name in devices:
        publish_status(bus, f"{COMPONENT}/{name}", "starting")
    results = await asyncio.gather(*(device.__aenter__() for device in devices.values()), return_exceptions=True)
    for (name, device), result in zip(devices.items(), results, strict=True):
        if isinstance(result, Exception):
            publish_status(bus, f"{COMPONENT}/{name}", "failed", _reason(result))
        elif not isinstance(result, BaseException):
            stack.push_async_exit(device)
            publish_status(bus, f"{COMPONENT}/{name}", "running")
    errors = [
        result
        for name, result in zip(devices, results, strict=True)
        if isinstance(result, BaseException) and not (name in optional and isinstance(result, Exception))
    ]
    if errors:
        raise errors[0]
    return {
        name: None if isinstance(result, BaseException) else result
        for name, result in zip(devices, results, strict=True)
    }

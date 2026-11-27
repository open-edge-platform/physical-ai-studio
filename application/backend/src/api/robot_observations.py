"""Joint observation stream for the robot detail page and form.

A ``RobotWorker`` reads the robot into a one-robot store and this socket streams
it. By default it is observe-only and never sends an action; with ``control``
the socket also takes joint targets from the client and the worker drives the
robot to them. A robot something else holds (a session or a calibration) is
refused with ``robot_device_already_owned``.
"""

import math
import time
from typing import Annotated
from uuid import UUID, uuid4

import anyio
from fastapi import APIRouter, Depends, Query, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import RobotClientFactoryDep, RuntimeSessionRegistryDep, get_robot_service
from exceptions import BaseException as AppBaseException
from runtime_session.store import ACTION, STATE, Store
from schemas.robot_stream import ActionCommand, ErrorEvent, ObservationEvent
from services import RobotService
from workers.base import WorkerStartupError
from workers.rate import Rate
from workers.robot_worker import RobotRecipe, RobotWorker, claim_robot

MAX_SPEED = 45.0
"""Joint units per second a manual target is approached at (degrees for most arms), so a jump cannot whip the arm."""

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])


def _error_payload(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, AppBaseException):
        return ErrorEvent(message=exc.message, error_code=exc.error_code).model_dump()
    message = exc.reason if isinstance(exc, WorkerStartupError) else str(exc)
    return ErrorEvent(
        message=message or "Failed to connect to the robot.",
        error_code="robot_connection_failed",
    ).model_dump()


async def _send_error_and_close(websocket: WebSocket, exc: BaseException) -> None:
    if isinstance(exc, AppBaseException):
        logger.warning("Robot observation websocket error: {} ({})", exc.message, exc.error_code)
    else:
        logger.exception("Unexpected error in robot observation websocket: {}", exc)
    try:
        await websocket.send_json(_error_payload(exc))
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
    except Exception as close_exc:
        logger.error("Could not close observation websocket after exception: {}", close_exc)


async def _stream_joint_observations(websocket: WebSocket, store: Store, worker: RobotWorker, fps: int) -> None:
    """Emit observation frames at ``fps``, with one connected-state after the first read."""
    sent_ready = False
    sent_at: float | None = None
    rate = Rate(fps, name="robot-observations")
    while True:
        if not worker.is_alive():
            raise RuntimeError(worker.failure or "The robot worker stopped.")
        sample = store.read(STATE)
        if sample is not None and sample.timestamp != sent_at:
            if not sent_ready:
                await websocket.send_json({"event": "state", "data": {"connected": True}})
                sent_ready = True
            await websocket.send_json(ObservationEvent(data=sample.named_values()).model_dump(mode="json"))
            sent_at = sample.timestamp
        await rate.sleep()


def _write_target(store: Store, target: dict[str, float]) -> None:
    """Merge ``target`` onto the current action and write it; dropped before the worker's first action."""
    current, state = store.read(ACTION), store.read(STATE)
    if current is None or state is None:
        return
    goal = current.value.copy()
    for name, value in target.items():
        goal[store.spec.joint_names.index(name)] = value
    # The worker eases toward an action stamped in the future: far enough out that no joint exceeds MAX_SPEED.
    store.write(ACTION, goal, timestamp=time.monotonic() + abs(goal - state.value).max() / MAX_SPEED)


async def _receive_commands(websocket: WebSocket, store: Store) -> None:
    """Write each valid ``action`` message from the client to the store; skip anything else."""
    names = set(store.spec.joint_names)
    while True:
        raw = await websocket.receive_text()
        try:
            command = ActionCommand.model_validate_json(raw)
        except ValueError as error:
            logger.warning("Ignoring invalid robot command: {}", error)
            continue
        unknown = command.data.keys() - names
        if unknown or not all(map(math.isfinite, command.data.values())):
            logger.warning("Ignoring robot command with unknown joints {} or non-finite values", sorted(unknown))
            continue
        _write_target(store, command.data)


@router.get(
    "/{robot_id}/observations/ws",
    tags=["WebSocket"],
    summary="Robot joint observations (WebSocket)",
    status_code=426,
)
async def robot_observations_websocket_openapi(
    project_id: UUID,  # noqa: ARG001
    robot_id: UUID,  # noqa: ARG001
    fps: Annotated[int, Query(ge=1, description="Display rate in frames per second")] = 30,  # noqa: ARG001
    control: Annotated[  # noqa: ARG001
        bool, Query(description="Also accept `{event: 'action', data: {joint: value}}` targets and drive the robot")
    ] = False,
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/{robot_id}/observations/ws")
async def robot_observations_websocket(  # noqa: PLR0913, PLR0917 - FastAPI injects each parameter
    websocket: WebSocket,
    project_id: UUID,
    robot_id: UUID,
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_client_factory: RobotClientFactoryDep,
    registry: RuntimeSessionRegistryDep,
    fps: Annotated[int, Query(ge=1)] = 30,
    control: Annotated[bool, Query()] = False,
) -> None:
    """Stream named joint positions; with ``control``, also drive the robot to the client's targets."""
    await websocket.accept()
    try:
        robot = await robot_service.get_robot_by_id(project_id, robot_id)
        registry.require_robot_free(robot.id, robot.name)
        driver, _definition = await robot_client_factory.build_robot_driver(robot, robot_client_factory)
        recipe = RobotRecipe.from_driver(robot.id, driver)
        with claim_robot(recipe):
            store = Store.create(recipe.spec, uuid4().hex)
            try:
                async with RobotWorker(recipe, store.name, store.spec, hz=fps, observe_only=not control) as worker:
                    if not control:
                        await _stream_joint_observations(websocket, store, worker, fps)
                    else:
                        # anyio, not asyncio.TaskGroup: Starlette cancels through anyio scopes, which asyncio leaks.
                        async with anyio.create_task_group() as tasks:
                            tasks.start_soon(_stream_joint_observations, websocket, store, worker, fps)
                            tasks.start_soon(_receive_commands, websocket, store)
            finally:
                store.unlink()
    except WebSocketDisconnect:
        pass
    except ExceptionGroup as group:  # from the control task group: either task ending ends both
        if not group.subgroup(WebSocketDisconnect):
            await _send_error_and_close(websocket, group.exceptions[0])
    except Exception as exc:
        await _send_error_and_close(websocket, exc)

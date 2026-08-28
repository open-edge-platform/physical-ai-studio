"""Read-only joint observation stream.

The robot detail page displays live joint positions and never sends actions.
Attaching a ``SharedRobot`` subscriber is enough: it joins the existing owner
when a runtime session is already driving the arm, and it never takes an
``rt-`` session lock.
"""

import asyncio
from typing import Annotated
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, Query, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from physicalai.robot import RobotError, SharedRobot

from api.dependencies import RobotClientFactoryDep, get_project_id, get_robot_id, get_robot_service
from exceptions import BaseException as AppBaseException
from robots.shared_robot_errors import translate_robot_error
from runtime.contract import ErrorEvent, ObservationEvent
from runtime.features import observation_to_dict
from services import RobotService
from workers.base import run_at_frequency

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])


def _error_payload(exc: BaseException) -> dict[str, str]:
    if isinstance(exc, AppBaseException):
        return ErrorEvent(message=exc.message, error_code=exc.error_code).model_dump()
    return ErrorEvent(
        message=str(exc) or "Failed to connect to the robot.",
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


async def _disconnect(shared_robot: SharedRobot | None) -> None:
    if shared_robot is None:
        return
    # Shield so a cancelled websocket task still drops the subscriber. Starlette
    # cancels the ASGI task on client close; an unshielded await in ``finally``
    # would skip disconnect and leak the SharedRobot attach.
    with anyio.CancelScope(shield=True):
        try:
            await asyncio.to_thread(shared_robot.disconnect)
        except Exception as exc:
            logger.warning("SharedRobot disconnect failed: {}", exc)


async def _stream_joint_observations(websocket: WebSocket, shared_robot: SharedRobot, fps: int) -> None:
    """Emit observation frames at ``fps``, with one connected-state after the first read."""
    sent_ready = False
    while True:
        async with run_at_frequency(fps):
            observation = await asyncio.to_thread(shared_robot.get_observation)
            data = observation_to_dict(
                shared_robot.joint_names,
                observation,
                include_velocities=False,
            )
            if not sent_ready:
                await websocket.send_json({"event": "state", "data": {"connected": True}})
                sent_ready = True
            await websocket.send_json(ObservationEvent(data=data).model_dump(mode="json"))


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
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/{robot_id}/observations/ws")
async def robot_observations_websocket(
    websocket: WebSocket,
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_client_factory: RobotClientFactoryDep,
    fps: Annotated[int, Query(ge=1)] = 30,
) -> None:
    """Stream named joint positions from a read-only SharedRobot attach."""
    await websocket.accept()
    shared_robot: SharedRobot | None = None
    robot_name: str | None = None
    try:
        robot = await robot_service.get_robot_by_id(project_id, robot_id)
        robot_name = robot.name
        shared_robot, _definition = await robot_client_factory.build_shared_robot(robot)
        await asyncio.to_thread(shared_robot.connect)
        await _stream_joint_observations(websocket, shared_robot, fps)
    except WebSocketDisconnect:
        pass
    except RobotError as exc:
        await _send_error_and_close(websocket, translate_robot_error(exc, robot_name=robot_name))
    except Exception as exc:
        await _send_error_and_close(websocket, exc)
    finally:
        await _disconnect(shared_robot)

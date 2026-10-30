"""Read-only joint observation stream for the robot detail page and form.

An observe-only ``RobotWorker`` reads the robot into a one-robot store and this
socket streams it. It never sends an action. A robot something else holds (a
session or a calibration) is refused with ``robot_device_already_owned``.
"""

from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import RobotClientFactoryDep, RuntimeSessionRegistryDep, get_robot_service
from exceptions import BaseException as AppBaseException
from runtime_session.store import STATE, Store
from schemas.robot_stream import ErrorEvent, ObservationEvent
from services import RobotService
from workers.base import WorkerStartupError
from workers.rate import Rate
from workers.robot_worker import RobotRecipe, RobotWorker, claim_robot

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
            data = dict(zip(sample.names, map(float, sample.value), strict=True))
            await websocket.send_json(ObservationEvent(data=data).model_dump(mode="json"))
            sent_at = sample.timestamp
        await rate.sleep()


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
    project_id: UUID,
    robot_id: UUID,
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_client_factory: RobotClientFactoryDep,
    registry: RuntimeSessionRegistryDep,
    fps: Annotated[int, Query(ge=1)] = 30,
) -> None:
    """Stream named joint positions from an observe-only robot worker."""
    await websocket.accept()
    try:
        robot = await robot_service.get_robot_by_id(project_id, robot_id)
        registry.require_robot_free(robot.id, robot.name)
        driver, _definition = await robot_client_factory.build_robot_driver(robot, robot_client_factory)
        recipe = RobotRecipe.from_driver(robot.id, driver)
        with claim_robot(recipe):
            store = Store.create(recipe.spec, uuid4().hex)
            try:
                async with RobotWorker(recipe, store.name, store.spec, hz=fps, observe_only=True) as worker:
                    await _stream_joint_observations(websocket, store, worker, fps)
            finally:
                store.unlink()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await _send_error_and_close(websocket, exc)

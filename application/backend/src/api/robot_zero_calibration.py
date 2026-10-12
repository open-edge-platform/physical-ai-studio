"""WebSocket endpoint for the robot zero-pose calibration wizard."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from loguru import logger

from api.dependencies import RobotCatalogServiceDep, RobotClientFactoryDep, get_project_id
from workers.robots.robot_zero_calibration_worker import RobotZeroCalibrationWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Setup"])


@router.get(
    "/zero-calibration/ws",
    tags=["WebSocket"],
    summary="Robot zero-pose calibration (WebSocket)",
    status_code=426,
)
async def robot_zero_calibration_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/zero-calibration/ws")
async def robot_zero_calibration_websocket(
    _project_id: Annotated[str, Depends(get_project_id)],
    robot_client_factory: RobotClientFactoryDep,
    catalog_service: RobotCatalogServiceDep,
    websocket: WebSocket,
) -> None:
    """Establish a WebSocket connection for the calibration wizard of a robot that is being added.

    The client sends ``{"command": "start", "robot": {...}}`` with the unsaved robot, then
    ``{"command": "set_zero"}`` once the arm is in its zero pose.
    """
    await websocket.accept()

    try:
        worker = RobotZeroCalibrationWorker(
            transport=WebSocketTransport(websocket),
            robot_client_factory=robot_client_factory,
            catalog_registry=catalog_service.registry,
        )

        await worker.run()

    except Exception as e:
        logger.exception(f"Unexpected error in robot calibration websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after error: {close_err}")

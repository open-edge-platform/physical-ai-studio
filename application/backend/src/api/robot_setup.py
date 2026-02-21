"""WebSocket endpoint for the SO101 robot setup wizard."""

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, status
from loguru import logger

from api.dependencies import get_project_id
from schemas.robot import RobotType
from workers.robots.so101_setup_worker import SO101SetupWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Setup"])


@router.websocket("/setup/ws")
async def robot_setup_websocket(
    project_id: Annotated[str, Depends(get_project_id)],
    websocket: WebSocket,
    robot_type: str,
    serial_number: str,
) -> None:
    """Establish a WebSocket connection for the SO101 robot setup wizard.

    Query parameters:
        robot_type: "SO101_Follower" or "SO101_Leader"
        serial_number: USB serial number of the robot's controller board
    """
    # Validate robot type
    if robot_type not in {RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER}:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": f"Unsupported robot type for setup: {robot_type}",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if not serial_number:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": "serial_number is required",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        worker = SO101SetupWorker(
            transport=WebSocketTransport(websocket),
            robot_type=robot_type,
            serial_number=serial_number,
        )

        await worker.run()

    except Exception as e:
        logger.exception(f"Unexpected error in robot setup websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after error: {close_err}")

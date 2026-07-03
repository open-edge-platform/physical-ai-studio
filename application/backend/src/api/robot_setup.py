"""WebSocket endpoint for the SO101 robot setup wizard."""

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, status
from loguru import logger

from api.dependencies import RobotConnectionManagerDep, get_project_id
from schemas import SerialPortInfo
from schemas.robot import RobotType
from workers.robots.so101_setup_worker import SO101SetupWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Robot Setup"])


@router.websocket("/setup/ws")
async def robot_setup_websocket(
    _project_id: Annotated[str, Depends(get_project_id)],
    robot_manager: RobotConnectionManagerDep,
    websocket: WebSocket,
    robot_type: str,
    serial_number: str | None = None,
    connection_string: str | None = None,
) -> None:
    """Establish a WebSocket connection for the SO101 robot setup wizard.

    Query parameters:
        robot_type: "SO101_Follower" or "SO101_Leader"
        serial_number: USB serial number of the robot's controller board (preferred)
        connection_string: serial port path (fallback when serial_number is unavailable)
    """
    # Validate robot type
    if robot_type not in {RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER}:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": f"Unsupported robot type for setup: {robot_type}",
                "error_code": "invalid_config",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    if not serial_number and not connection_string:
        await websocket.accept()
        await websocket.send_json(
            {
                "event": "error",
                "message": "Either serial_number or connection_string is required",
                "error_code": "invalid_config",
            }
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()

    try:
        serial_port = SerialPortInfo(
            connection_string=connection_string,
            serial_number=serial_number,
        )
        worker = SO101SetupWorker(
            transport=WebSocketTransport(websocket),
            robot_type=robot_type,
            serial_port=serial_port,
            robot_manager=robot_manager,
        )

        await worker.run()

    except Exception as e:
        logger.exception(f"Unexpected error in robot setup websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after error: {close_err}")

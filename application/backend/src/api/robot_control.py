from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from loguru import logger

from api.dependencies import (
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    RobotRegistryDep,
    get_project_id,
    get_robot_id,
    get_robot_service,
)
from services import RobotService
from workers.robots.robot_worker import RobotWorker
from workers.transport.websocket_transport import WebSocketTransport

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


@router.get("/{robot_id}/ws", tags=["WebSocket"], summary="Robot control (WebSocket)", status_code=426)
async def robot_websocket_openapi(project_id: UUID, robot_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/{robot_id}/ws")
async def robot_websocket(  # noqa: PLR0913
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    websocket: WebSocket,
    registry: RobotRegistryDep,
    normalize: bool = True,
    fps: int = 30,
) -> None:
    """
    Establish a WebSocket connection for real-time robot state monitoring and control.

    Args:
        project_id: ID of the project.
        robot_id: ID of the robot.
        robot_service: Service for robot metadata.
        robot_manager: Connection manager for robot discovery.
        calibration_service: Service for loading calibration.
        websocket: The FastAPI WebSocket instance.
        registry: Registry for managing active robot workers.
        normalize: Whether to use normalized joint values.
        fps: Target frequency for state updates.
    """
    await websocket.accept()

    robot = await robot_service.get_robot_by_id(project_id, robot_id)
    logger.info("Found robot with websocket {}", robot)

    worker_id = uuid4()

    try:
        # Create worker
        worker = RobotWorker(
            robot,
            WebSocketTransport(websocket),
            # Manager and calibration service are used to create robot config
            robot_manager,
            calibration_service,
            # Read configuration
            fps,
            normalize,
        )

        # Register worker
        await registry.create_and_register(worker_id, worker)

        # Run worker (blocks until complete)
        await worker.run()

    except ValueError as e:
        logger.error(f"Failed to register worker: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        except Exception as close_err:
            logger.error(f"Could not close websocket after ValueError: {close_err}")

    except Exception as e:
        logger.exception(f"Unexpected error in robot websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after Exception: {close_err}")

    finally:
        # Unregister worker
        await registry.unregister(worker_id)

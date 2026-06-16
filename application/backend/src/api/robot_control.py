from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import (
    RobotCalibrationServiceDep,
    RobotConnectionManagerDep,
    SchedulerDep,
    get_project_id,
    get_robot_id,
    get_robot_service,
)
from robots.robot_client_factory import RobotClientFactory
from services import RobotService
from workers.base import run_at_frequency
from workers.teleoperate_worker import TeleoperateWorker

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


@router.get("/ws", tags=["WebSocket"], summary="Robot control (WebSocket)", status_code=426)
async def robot_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def robot_websocket(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_manager: RobotConnectionManagerDep,
    calibration_service: RobotCalibrationServiceDep,
    websocket: WebSocket,
    scheduler: SchedulerDep,
    fps: int = 30,
) -> None:
    """
    Establish a WebSocket connection for real-time robot state monitoring and control.

    Args:
        project_id: ID of the project.
        robot_service: Service for robot metadata.
        robot_manager: Connection manager for robot discovery.
        calibration_service: Service for loading calibration.
        websocket: The FastAPI WebSocket instance.
        registry: Registry for managing active robot workers.
        normalize: Whether to use normalized joint values.
        fps: Target frequency for state updates.
    """
    await websocket.accept()
    worker = None
    try:
        settings = await websocket.receive_json("text")
        follower_id = get_robot_id(settings["follower_id"])
        robot_client_factory = RobotClientFactory(robot_manager, calibration_service)
        follower = await robot_service.get_robot_by_id(project_id, follower_id)
        follower_client = await robot_client_factory.build(follower)

        leader_client = None
        if "leader_id" in settings:
            leader_id = get_robot_id(settings["leader_id"])
            leader = await robot_service.get_robot_by_id(project_id, leader_id)
            leader_client = await robot_client_factory.build(leader)

        # Create worker
        worker = TeleoperateWorker(
            follower=follower_client, leader=leader_client, frequency=fps, mp_stop_event=scheduler.mp_stop_event
        )
        worker.start()
        while True:
            action_keys = follower_client.features()
            async with run_at_frequency(fps):
                raw_state = worker.get_state()
                observation: dict[str, Any] = {i: raw_state[k] for k, i in enumerate(action_keys)}
                await websocket.send_json({"event": "state_was_updated", "state": observation, "is_controlled": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"Unexpected error in robot websocket: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after Exception: {close_err}")

    finally:
        if worker:
            worker.stop()

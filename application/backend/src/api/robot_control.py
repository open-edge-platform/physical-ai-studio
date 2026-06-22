import asyncio
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


def _build_robot_control_state(worker: TeleoperateWorker) -> dict:
    return {"connected": worker.loaded_event.is_set(), "follower_source": worker.get_action_read_state()}


async def handle_outgoing(
    websocket: WebSocket, worker: TeleoperateWorker, features: list[str], update_frequency: int
) -> None:
    """Handle outgoing messages from teleoperate worker."""
    try:
        while not worker.should_stop():
            async with run_at_frequency(update_frequency):
                raw_state = worker.get_state()
                observation: dict[str, Any] = {i: raw_state[k] for k, i in enumerate(features)}
                await websocket.send_json({"event": "observation", "data": observation})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Outgoing task stopped: {e}")


async def handle_incoming(websocket: WebSocket, worker: TeleoperateWorker) -> None:
    """Handle incoming messages from client to teleoperate worker."""
    try:
        while not worker.should_stop():
            data = await websocket.receive_json("text")
            payload = data.get("data", {})
            match data["event"]:
                case "set_follower_source":
                    worker.set_action_read_state(payload)
            await websocket.send_json({"event": "state", "data": _build_robot_control_state(worker)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Incoming task stopped: {type(e).__name__} - {e}")
        logger.info("Except: disconnected!")


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
        features = follower_client.features()

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

        await asyncio.to_thread(worker.loaded_event.wait)
        await websocket.send_json({"event": "state", "data": _build_robot_control_state(worker)})

        incoming_task = asyncio.create_task(handle_incoming(websocket, worker))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, worker, features, fps))

        _, pending = await asyncio.wait(
            {incoming_task, outgoing_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
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

import asyncio
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger

from api.dependencies import RobotClientFactoryDep, SchedulerDep, get_project_id, get_robot_id, get_robot_service
from exceptions import BaseException as AppBaseException
from exceptions import RobotPluginUnavailableError
from schemas.robot import ReadableRobot, UnavailableRobot
from services import RobotService
from workers.base import run_at_frequency
from workers.teleoperate_worker import TeleoperateWorker

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


def _websocket_error_payload(exc: Exception) -> dict[str, str]:
    if isinstance(exc, AppBaseException):
        return {"event": "error", "message": exc.message, "error_code": exc.error_code}
    return {
        "event": "error",
        "message": str(exc) or "Failed to connect to the robot.",
        "error_code": "robot_connection_failed",
    }


@router.get("/ws", tags=["WebSocket"], summary="Robot control (WebSocket)", status_code=426)
async def robot_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


def _build_robot_control_state(worker: TeleoperateWorker) -> dict:
    return {"connected": worker.loaded_event.is_set(), "follower_source": worker.get_action_read_state()}


def _ensure_robot_available(robot: ReadableRobot) -> None:
    if isinstance(robot, UnavailableRobot):
        raise RobotPluginUnavailableError(robot.name, robot.type)


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
    robot_client_factory: RobotClientFactoryDep,
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
        follower = await robot_service.get_robot_by_id(project_id, follower_id)
        _ensure_robot_available(follower)
        leader = None
        if "leader_id" in settings:
            leader_id = get_robot_id(settings["leader_id"])
            leader = await robot_service.get_robot_by_id(project_id, leader_id)
            _ensure_robot_available(leader)

        # Create worker
        worker = TeleoperateWorker(
            robot_client_factory=robot_client_factory,
            follower=follower,
            leader=leader,
            frequency=fps,
            stop_event=scheduler.mp_stop_event,
        )
        worker.start()

        await worker.wait_until_loaded()
        features = worker.features
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
        if isinstance(e, AppBaseException):
            logger.warning("Robot websocket error: {} ({})", e.message, e.error_code)
        else:
            logger.exception(f"Unexpected error in robot websocket: {e}")
        try:
            await websocket.send_json(_websocket_error_payload(e))
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_err:
            logger.error(f"Could not close websocket after Exception: {close_err}")

    finally:
        if worker:
            worker.stop()

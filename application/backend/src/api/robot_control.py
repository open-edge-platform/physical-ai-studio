import asyncio
import queue
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from api.dependencies import RobotClientFactoryDep, SchedulerDep, get_project_id, get_robot_id, get_robot_service
from exceptions import BaseException as AppBaseException
from exceptions import RobotPluginUnavailableError
from runtime.config_builder import build_runtime_config
from runtime.contract import DisconnectCommand, QueueEventSink, SetFollowerSourceCommand
from runtime.hosts.thread_host import RuntimeThreadHost
from runtime.session import RuntimeSession
from schemas.robot import ReadableRobot, UnavailableRobot
from services import RobotService

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])


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


def _ensure_robot_available(robot: ReadableRobot) -> None:
    if isinstance(robot, UnavailableRobot):
        raise RobotPluginUnavailableError(robot.name, robot.type)


async def handle_outgoing(websocket: WebSocket, event_sink: QueueEventSink, host: RuntimeThreadHost) -> None:
    """Send all runtime events from one task so websocket writes cannot overlap."""
    try:
        while True:
            try:
                event = event_sink.get_nowait()
            except queue.Empty:
                if host.completed.is_set():
                    if host.error is not None:
                        raise host.error
                    return
                await asyncio.sleep(0.01)
                continue
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass


async def handle_incoming(websocket: WebSocket, session: RuntimeSession) -> None:
    """Validate websocket commands and apply them to the runtime mailbox."""
    try:
        while True:
            message = await websocket.receive_json("text")
            if message.get("event") != "set_follower_source":
                continue
            payload = message.get("data", {})
            try:
                command = SetFollowerSourceCommand.model_validate({"follower_source": payload.get("follower_source")})
            except ValidationError as exc:
                logger.warning("Rejected malformed set_follower_source payload {}: {}", payload, exc)
                continue
            session.apply(command)
    except WebSocketDisconnect:
        session.apply(DisconnectCommand())


@router.websocket("/ws")
async def robot_websocket(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_client_factory: RobotClientFactoryDep,
    websocket: WebSocket,
    scheduler: SchedulerDep,
    fps: int = 30,
) -> None:
    """Stream follower state and accept hold/teleop mode changes."""
    await websocket.accept()
    host: RuntimeThreadHost | None = None
    session: RuntimeSession | None = None
    try:
        settings = await websocket.receive_json("text")
        follower_id = get_robot_id(settings["follower_id"])
        follower = await robot_service.get_robot_by_id(project_id, follower_id)
        _ensure_robot_available(follower)
        leader = None
        if settings.get("leader_id") is not None:
            leader_id = get_robot_id(settings["leader_id"])
            leader = await robot_service.get_robot_by_id(project_id, leader_id)
            _ensure_robot_available(leader)

        document = await build_runtime_config(
            follower=follower,
            leader=leader,
            cameras=[],
            fps=fps,
            robot_factory=robot_client_factory,
        )
        event_sink = QueueEventSink()
        session = RuntimeSession(
            document,
            event_sink=event_sink,
            follower_name=follower.name,
            leader_name=None if leader is None else leader.name,
        )
        host = RuntimeThreadHost(session, stop_event=scheduler.mp_stop_event)
        host.start()
        await host.wait_until_ready()

        incoming_task = asyncio.create_task(handle_incoming(websocket, session))
        outgoing_task = asyncio.create_task(handle_outgoing(websocket, event_sink, host))
        done, pending = await asyncio.wait(
            {incoming_task, outgoing_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if isinstance(exc, AppBaseException):
            logger.warning("Robot websocket error: {} ({})", exc.message, exc.error_code)
        else:
            logger.exception("Unexpected error in robot websocket: {}", exc)
        try:
            await websocket.send_json(_websocket_error_payload(exc))
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception as close_exc:
            logger.error("Could not close websocket after exception: {}", close_exc)
    finally:
        if session is not None:
            session.apply(DisconnectCommand())
        if host is not None:
            await asyncio.to_thread(host.stop)

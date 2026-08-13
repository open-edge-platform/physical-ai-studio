import asyncio
import queue
import time
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from api.dependencies import RobotClientFactoryDep, get_project_id, get_robot_id, get_robot_service
from exceptions import BaseException as AppBaseException
from exceptions import RobotPluginUnavailableError
from runtime.config_builder import build_runtime_config
from runtime.contract import DisconnectCommand, SetFollowerSourceCommand
from runtime.hosts.process_host import RuntimeProcessHost
from runtime.transport.client import RuntimeProcessError, RuntimeSessionClient
from runtime.transport.ids import runtime_session_name
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


async def handle_outgoing(
    websocket: WebSocket,
    client: RuntimeSessionClient,
    host: RuntimeProcessHost,
) -> None:
    """Send all runtime events from one task so websocket writes cannot overlap."""
    process_dead_since: float | None = None
    try:
        while True:
            try:
                event = client.get_nowait()
            except queue.Empty:
                if client.error is not None:
                    raise client.error
                if not host.is_alive():
                    if client.shutdown_received or host.exited_cleanly:
                        return
                    if host.error is not None:
                        raise host.error
                    process_dead_since = process_dead_since or time.monotonic()
                    if time.monotonic() - process_dead_since >= 0.2:
                        raise RuntimeProcessError("Runtime session process stopped unexpectedly")
                await asyncio.sleep(0.01)
                continue
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass


async def handle_incoming(websocket: WebSocket, client: RuntimeSessionClient) -> None:
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
            client.apply(command)
    except WebSocketDisconnect:
        client.apply(DisconnectCommand())


async def start_runtime_session(client: RuntimeSessionClient, host: RuntimeProcessHost) -> None:
    """Spawn, then wait for transport and hardware readiness, off the event loop."""
    await asyncio.to_thread(host.start)
    await asyncio.to_thread(client.connect, process=host)
    await asyncio.to_thread(client.wait_until_ready, host)


@router.websocket("/ws")
async def robot_websocket(  # noqa: PLR0912, PLR0915
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot_client_factory: RobotClientFactoryDep,
    websocket: WebSocket,
    fps: int = 30,
) -> None:
    """Stream follower state and accept hold/teleop mode changes."""
    await websocket.accept()
    host: RuntimeProcessHost | None = None
    client: RuntimeSessionClient | None = None
    incoming_task: asyncio.Task[None] | None = None
    outgoing_task: asyncio.Task[None] | None = None
    startup_task: asyncio.Task[None] | None = None
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
        session_name = runtime_session_name(follower.id)
        client = RuntimeSessionClient(session_name)
        await asyncio.to_thread(client.open)
        host = RuntimeProcessHost(
            session_name,
            document,
            follower_name=follower.name,
            leader_name=None if leader is None else leader.name,
        )
        incoming_task = asyncio.create_task(handle_incoming(websocket, client))
        startup_task = asyncio.create_task(start_runtime_session(client, host))
        done, _ = await asyncio.wait(
            {incoming_task, startup_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if incoming_task in done:
            incoming_task.result()
            await asyncio.to_thread(host.stop)
            await asyncio.gather(startup_task, return_exceptions=True)
            return
        startup_task.result()

        outgoing_task = asyncio.create_task(handle_outgoing(websocket, client, host))
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
        tasks = [task for task in (incoming_task, outgoing_task, startup_task) if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if client is not None:
            client.apply(DisconnectCommand())
        if host is not None:
            await asyncio.to_thread(host.stop)
        if client is not None:
            client.close()

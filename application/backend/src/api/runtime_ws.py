from __future__ import annotations

import asyncio
import queue
import time
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID  # noqa: TC003  # FastAPI evaluates websocket annotations at runtime

from fastapi import APIRouter, Depends, WebSocket, status
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError

from api.dependencies import (
    CameraClaimRegistryDep,
    ProjectCameraServiceDep,
    ProjectServiceDep,
    RobotClientFactoryDep,
    RuntimeSessionRegistryDep,
    get_camera_id,
    get_project_id,
    get_robot_id,
    get_robot_service,
)
from exceptions import BaseException as AppBaseException
from exceptions import RobotPluginUnavailableError
from runtime.config_builder import RUNTIME_FPS, build_runtime_config
from runtime.contract import CommandAdapter
from runtime.handle import RuntimeProcessError, RuntimeSessionHandle
from runtime.ids import runtime_session_name
from schemas.robot import ReadableRobot, UnavailableRobot
from services import ProjectCameraService, RobotService
from services.camera_claims import CameraClaim, CameraClaimRegistry, settings_from_camera

if TYPE_CHECKING:
    from runtime.registry import RuntimeSessionRegistry
    from schemas.project_camera import Camera
    from schemas.robot import Robot

_stopping_sessions: set[asyncio.Task[None]] = set()

router = APIRouter(prefix="/api/projects/{project_id}/runtime", tags=["Runtime"])


def _websocket_error_payload(exc: Exception) -> dict[str, str]:
    if isinstance(exc, AppBaseException):
        return {"event": "error", "message": exc.message, "error_code": exc.error_code}
    return {
        "event": "error",
        "message": str(exc) or "Failed to connect to the robot.",
        "error_code": "robot_connection_failed",
    }


def _ensure_robot_available(robot: ReadableRobot) -> None:
    if isinstance(robot, UnavailableRobot):
        raise RobotPluginUnavailableError(robot.name, robot.type)


async def handle_outgoing(websocket: WebSocket, handle: RuntimeSessionHandle) -> None:
    """Send all runtime events from one task so websocket writes cannot overlap."""
    process_dead_since: float | None = None
    try:
        while True:
            try:
                event = handle.get_nowait()
            except queue.Empty:
                if handle.error is not None:
                    raise handle.error from None
                if not handle.is_alive():
                    if handle.stopping:
                        return
                    # The child flushes its queue on exit; give a fatal event a moment to arrive.
                    process_dead_since = process_dead_since or time.monotonic()
                    if time.monotonic() - process_dead_since >= 0.2:
                        raise RuntimeProcessError("Runtime session process stopped unexpectedly") from None
                await asyncio.sleep(0.01)
                continue
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass


_RUNTIME_PUBLICATIONS = frozenset(
    {
        "set_follower_source",
        "load_model",
        "start_task",
        "stop_task",
        "load_dataset",
        "start_recording",
    }
)
_RUNTIME_REQUESTS = frozenset({"save_episode", "discard_episode"})


async def handle_incoming(websocket: WebSocket, handle: RuntimeSessionHandle) -> None:
    """Validate websocket commands and forward them to the session worker.

    Returns on an explicit ``disconnect`` event; a closed socket ends the same
    way. Either ends the session.
    """
    try:
        while True:
            message = await websocket.receive_json("text")
            event = message.get("event")
            if event == "disconnect":
                return
            if event not in _RUNTIME_PUBLICATIONS and event not in _RUNTIME_REQUESTS:
                continue
            payload = dict(message.get("data") or {})
            payload["command"] = event
            if message.get("request_id") is not None:
                payload["request_id"] = message["request_id"]
            try:
                command = CommandAdapter.validate_python(payload)
            except ValidationError as exc:
                logger.warning("Rejected malformed {} payload {}: {}", event, payload, exc)
                continue
            # Requests are answered by an ack on the event stream.
            handle.apply(command)
    except WebSocketDisconnect:
        logger.debug("Robot control websocket closed; stopping the runtime session")


async def start_runtime_session(handle: RuntimeSessionHandle) -> None:
    """Spawn the worker, then wait for hardware readiness, off the event loop."""
    await asyncio.to_thread(handle.start)
    await asyncio.to_thread(handle.wait_until_ready)


async def _devices_from_handshake(
    handshake: dict[str, Any],
    project_id: UUID,
    robot_service: RobotService,
    camera_service: ProjectCameraService,
) -> tuple[Robot, Robot | None, list[Camera]]:
    """Resolve handshake ids against the project so a client cannot name another project's devices."""
    follower_id = get_robot_id(handshake["follower_id"])
    follower = await robot_service.get_robot_by_id(project_id, follower_id)
    _ensure_robot_available(follower)
    leader = None
    if handshake.get("leader_id") is not None:
        leader_id = get_robot_id(handshake["leader_id"])
        leader = await robot_service.get_robot_by_id(project_id, leader_id)

        if isinstance(leader, UnavailableRobot):
            leader = None

    raw_camera_ids = handshake.get("camera_ids") or []
    if not isinstance(raw_camera_ids, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="camera_ids must be a list")
    cameras: list[Camera] = []
    for raw in raw_camera_ids:
        if not isinstance(raw, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID")
        cameras.append(await camera_service.get_camera_by_id(project_id, get_camera_id(raw)))
    return follower, leader, cameras


async def _stop_session(
    handle: RuntimeSessionHandle,
    sessions: RuntimeSessionRegistry,
    claims: CameraClaimRegistry,
    *,
    registered: bool,
    claim_generation: int | None,
) -> None:
    """Stop the worker, then free the follower and camera claims it held."""
    try:
        await asyncio.to_thread(handle.stop)
    finally:
        if registered:
            sessions.release(handle)
        if claim_generation is not None:
            claims.release(handle.session_name, generation=claim_generation)


def _camera_claims(
    *,
    cameras: list[Camera],
    session_name: str,
    project_id: UUID,
    project_name: str,
) -> list[CameraClaim]:
    claims: list[CameraClaim] = []
    for camera in cameras:
        if camera.fingerprint is None:
            raise ValueError(f"Camera {camera.name!r} must be reselected")
        claims.append(
            CameraClaim(
                fingerprint=camera.fingerprint,
                settings=settings_from_camera(camera),
                holder=session_name,
                project_id=project_id,
                project_name=project_name,
            )
        )
    return claims


@router.get("/ws", tags=["WebSocket"], summary="Runtime session (WebSocket)", status_code=426)
async def runtime_websocket_openapi(project_id: UUID) -> Response:  # noqa: ARG001
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtime_websocket(  # noqa: PLR0913, PLR0915
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    camera_service: ProjectCameraServiceDep,
    project_service: ProjectServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    claims: CameraClaimRegistryDep,
    sessions: RuntimeSessionRegistryDep,
    websocket: WebSocket,
) -> None:
    """Run a runtime session for as long as this websocket is open."""
    await websocket.accept()
    handle: RuntimeSessionHandle | None = None
    registered = False
    incoming_task: asyncio.Task[None] | None = None
    outgoing_task: asyncio.Task[None] | None = None
    startup_task: asyncio.Task[None] | None = None
    claim_generation: int | None = None
    try:
        handshake = await websocket.receive_json("text")
        follower, leader, cameras = await _devices_from_handshake(handshake, project_id, robot_service, camera_service)
        document = await build_runtime_config(
            follower=follower,
            leader=leader,
            cameras=cameras,
            fps=RUNTIME_FPS,
            robot_factory=robot_client_factory,
        )
        name = runtime_session_name(follower.id)
        project = await project_service.get_project_by_id(project_id)
        handle = RuntimeSessionHandle(
            name,
            follower_id=follower.id,
            document=document,
            follower_name=follower.name,
            leader_name=None if leader is None else leader.name,
            stop_event=sessions.stop_event,
        )
        await sessions.acquire(handle)
        registered = True
        claim_generation = claims.claim(
            _camera_claims(
                cameras=cameras,
                session_name=name,
                project_id=project_id,
                project_name=project.name,
            )
        )

        incoming_task = asyncio.create_task(handle_incoming(websocket, handle))
        startup_task = asyncio.create_task(start_runtime_session(handle))
        done, _ = await asyncio.wait({incoming_task, startup_task}, return_when=asyncio.FIRST_COMPLETED)
        if incoming_task in done:
            # Closed or disconnected during startup; the finally block stops the worker.
            incoming_task.result()
            return
        startup_task.result()

        outgoing_task = asyncio.create_task(handle_outgoing(websocket, handle))
        done, pending = await asyncio.wait({incoming_task, outgoing_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if isinstance(exc, AppBaseException):
            logger.warning("Runtime websocket error: {} ({})", exc.message, exc.error_code)
        else:
            logger.exception("Unexpected error in runtime websocket: {}", exc)
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
        if handle is not None:
            # Shielded: teardown finalizes the recording and releases the arm,
            # and the robot must stay claimed until it has, even if this
            # handler is cancelled.
            stop_task = asyncio.create_task(
                _stop_session(
                    handle,
                    sessions,
                    claims,
                    registered=registered,
                    claim_generation=claim_generation,
                )
            )
            _stopping_sessions.add(stop_task)
            stop_task.add_done_callback(_stopping_sessions.discard)
            await asyncio.shield(stop_task)

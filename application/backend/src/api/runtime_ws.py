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
    SettingsDep,
    get_camera_id,
    get_project_id,
    get_robot_id,
    get_robot_service,
)
from exceptions import BaseException as AppBaseException
from exceptions import RobotPluginUnavailableError
from runtime.config_builder import RUNTIME_FPS, build_runtime_config
from runtime.contract import CommandAdapter, DisconnectCommand
from runtime.owner import RuntimeSessionOwner
from runtime.session import RECORDING_TEARDOWN_TIMEOUT_S
from runtime.transport.client import RuntimeProcessError, RuntimeSessionClient
from runtime.transport.ids import runtime_session_name
from schemas.robot import ReadableRobot, UnavailableRobot
from services import ProjectCameraService, RobotService
from services.camera_claims import CameraClaim, CameraClaimRegistry, settings_from_camera

if TYPE_CHECKING:
    from schemas.project_camera import Camera
    from schemas.robot import Robot

_CLAIM_POLL_INTERVAL_S = 0.5
_claim_waiters: set[asyncio.Task[None]] = set()

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


async def handle_outgoing(
    websocket: WebSocket,
    client: RuntimeSessionClient,
    owner: RuntimeSessionOwner,
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
                if not owner.is_alive():
                    if client.shutdown_received or owner.exited_cleanly():
                        return
                    if owner.error is not None:
                        raise owner.error
                    process_dead_since = process_dead_since or time.monotonic()
                    if time.monotonic() - process_dead_since >= 0.2:
                        raise RuntimeProcessError("Runtime session process stopped unexpectedly")
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


async def handle_incoming(websocket: WebSocket, client: RuntimeSessionClient) -> None:
    """Validate websocket commands and apply them to the runtime mailbox."""
    try:
        while True:
            message = await websocket.receive_json("text")
            event = message.get("event")
            if event == "disconnect":
                client.apply(DisconnectCommand(request_id=message.get("request_id")))
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
            if event in _RUNTIME_REQUESTS:
                ack = await asyncio.to_thread(client.request, command, RECORDING_TEARDOWN_TIMEOUT_S)
                client.deliver(ack)
                continue
            client.apply(command)
    except WebSocketDisconnect:
        # Browser close / refresh is a detach. The child stays up until an
        # explicit disconnect event or idle timeout.
        logger.debug("Robot control websocket closed; detaching from the runtime session")


async def start_runtime_session(
    client: RuntimeSessionClient, owner: RuntimeSessionOwner, *, replace: bool = False
) -> None:
    """Attach or spawn, then wait for hardware readiness, off the event loop.

    ``replace`` stops any live session for this follower before spawning with
    the handshake's current recipe.
    """
    await asyncio.to_thread(owner.connect, replace=replace)
    await asyncio.to_thread(client.wait_until_ready, owner)


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
        _ensure_robot_available(leader)

    raw_camera_ids = handshake.get("camera_ids") or []
    if not isinstance(raw_camera_ids, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="camera_ids must be a list")
    cameras: list[Camera] = []
    for raw in raw_camera_ids:
        if not isinstance(raw, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid camera ID")
        cameras.append(await camera_service.get_camera_by_id(project_id, get_camera_id(raw)))
    return follower, leader, cameras


async def _release_claims_when_dead(
    owner: RuntimeSessionOwner,
    claims: CameraClaimRegistry,
    holder: str,
    generation: int,
) -> None:
    """Drop pins after the detached session process exits.

    A websocket close is a detach, not a stop. Releasing immediately would let
    a preview reconfigure cameras mid-recording. Ignore a stale generation so
    a reconnect that reused this holder keeps its pin.
    """
    try:
        # Poll the child; there is no in-process Event for process death.
        while await asyncio.to_thread(owner.is_alive):  # noqa: ASYNC110
            await asyncio.sleep(_CLAIM_POLL_INTERVAL_S)
    except asyncio.CancelledError:
        return
    claims.release(holder, generation=generation)


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
async def runtime_websocket(  # noqa: PLR0913, PLR0915, PLR0912
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    camera_service: ProjectCameraServiceDep,
    project_service: ProjectServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    claims: CameraClaimRegistryDep,
    settings: SettingsDep,
    websocket: WebSocket,
) -> None:
    """Stream follower state and accept hold/teleop mode changes."""
    await websocket.accept()
    owner: RuntimeSessionOwner | None = None
    client: RuntimeSessionClient | None = None
    incoming_task: asyncio.Task[None] | None = None
    outgoing_task: asyncio.Task[None] | None = None
    startup_task: asyncio.Task[None] | None = None
    session_name: str | None = None
    claimed = False
    claim_generation = 0
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
        session_name = name
        project = await project_service.get_project_by_id(project_id)
        claim_generation = claims.claim(
            _camera_claims(
                cameras=cameras,
                session_name=name,
                project_id=project_id,
                project_name=project.name,
            )
        )
        claimed = True
        client = RuntimeSessionClient(name)
        await asyncio.to_thread(client.open)
        owner = RuntimeSessionOwner(
            client,
            session_name=name,
            document=document,
            follower_name=follower.name,
            leader_name=None if leader is None else leader.name,
            idle_timeout_s=settings.runtime_idle_timeout_s,
        )
        incoming_task = asyncio.create_task(handle_incoming(websocket, client))
        try:
            startup_task = asyncio.create_task(
                start_runtime_session(client, owner, replace=handshake.get("restart") is True)
            )
            done, _ = await asyncio.wait(
                {incoming_task, startup_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:
            claims.release(name, generation=claim_generation)
            claimed = False
            raise
        if incoming_task in done:
            incoming_task.result()
            # Interrupt a spawn that is not yet discoverable. Once /metadata is
            # up this close is a detach — same as after startup.
            await asyncio.to_thread(owner.stop_abandoned_spawn)
            await asyncio.gather(startup_task, return_exceptions=True)
            return
        startup_task.result()

        outgoing_task = asyncio.create_task(handle_outgoing(websocket, client, owner))
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
        if client is not None:
            client.close()
        if claimed and session_name is not None:
            if owner is None or not owner.is_alive():
                claims.release(session_name, generation=claim_generation)
            else:
                waiter = asyncio.create_task(
                    _release_claims_when_dead(owner, claims, session_name, claim_generation),
                    name=f"release-claims-{session_name}",
                )
                _claim_waiters.add(waiter)
                waiter.add_done_callback(_claim_waiters.discard)

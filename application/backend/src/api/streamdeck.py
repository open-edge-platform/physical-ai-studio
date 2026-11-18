"""UI-managed recording attachments for a Stream Deck helper on the USB host."""

from ipaddress import ip_address
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Header, Request

from api.dependencies import RuntimeSessionRegistryDep, StreamDeckServiceDep
from exceptions import BaseException as APIError
from exceptions import ResourceNotFoundError, ResourceType
from runtime_session.commands import request_command
from schemas.runtime_session import AckMessage, SetControlCommand
from schemas.streamdeck import (
    StreamDeckHeartbeat,
    StreamDeckInferenceCommand,
    StreamDeckStatus,
    StreamDeckTask,
    StreamDeckText,
    StreamDeckTranscript,
)

router = APIRouter(prefix="/api/streamdeck", tags=["Stream Deck"])


@router.get("")
async def streamdeck_status(
    request: Request, service: StreamDeckServiceDep, registry: RuntimeSessionRegistryDep
) -> StreamDeckStatus:
    """Show helper availability and whether this local backend can start it."""
    return service.status(registry).model_copy(
        update={
            "can_start_native": _local_request(request) and service.native_launch_available(),
        }
    )


def _local_request(request: Request) -> bool:
    def loopback(host: str | None) -> bool:
        if host == "localhost":
            return True
        try:
            return host is not None and ip_address(host).is_loopback
        except ValueError:
            return False

    origin = request.headers.get("origin")
    try:
        origin_host = urlsplit(origin).hostname if origin is not None else None
    except ValueError:
        return False
    return (
        request.client is not None
        and loopback(request.client.host)
        and loopback(request.url.hostname)
        and (origin is None or loopback(origin_host))
    )


@router.post("/start")
async def start_streamdeck_helper(
    request: Request,
    service: StreamDeckServiceDep,
    registry: RuntimeSessionRegistryDep,
    x_streamdeck_launch: Annotated[Literal["1"], Header()],  # noqa: ARG001 - required and validated by FastAPI
) -> StreamDeckStatus:
    """Launch only for a local UI gesture; the custom header also prevents simple-form CSRF."""
    if not _local_request(request):
        raise APIError("Only the local Studio UI can launch a native helper.", "streamdeck_launch_forbidden", 403)
    if not service.status(registry).helper_online:
        try:
            service.start()
        except OSError as error:
            raise APIError(
                "Could not start the helper. Check uv and the Studio log directory.", "streamdeck_launch_failed", 503
            ) from error
    return service.status(registry)


@router.post("/heartbeat")
async def streamdeck_heartbeat(
    heartbeat: StreamDeckHeartbeat, service: StreamDeckServiceDep, registry: RuntimeSessionRegistryDep
) -> StreamDeckStatus:
    """Renew the host helper's lease and return the desired workspace attachment."""
    return service.report(heartbeat, registry)


@router.put("/sessions/{session_id}")
async def connect_streamdeck(
    session_id: str, task: StreamDeckTask, service: StreamDeckServiceDep, registry: RuntimeSessionRegistryDep
) -> StreamDeckStatus:
    """Request USB access from the helper; repeat to synchronize this workspace's task text."""
    session = registry.get(session_id)
    if session is None or session.closed.is_set():
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    return service.connect(session_id, task.task, registry, page=task.page, inference=task.inference)


@router.post("/sessions/{session_id}/inference")
async def control_inference(
    session_id: str,
    command: StreamDeckInferenceCommand,
    service: StreamDeckServiceDep,
    registry: RuntimeSessionRegistryDep,
    x_streamdeck_connection: Annotated[str, Header()],
) -> AckMessage:
    """Start with the workspace's settings, or hold the robot; never accept arbitrary controls."""
    target = service.require_target(session_id, x_streamdeck_connection, registry, page="inference")
    session = registry.get(session_id)
    if session is None:
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    current = session.state().control
    if command.action == "start":
        if current is not None or service.status(registry).dictation_status in {
            "preparing",
            "listening",
            "transcribing",
        }:
            raise APIError("Stop control and finish dictation before starting inference.", "control_active", 409)
        control = target.inference.model_copy(update={"task": target.task})
    else:
        if current is None:
            return AckMessage(request_id=command.request_id, ok=True)
        if current.type != "model":
            raise APIError("The running control is not inference.", "control_active", 409)
        control = None
    return await request_command(
        session.bus, SetControlCommand(request_id=command.request_id, control=control).model_dump_json()
    )


@router.post("/sessions/{session_id}/dictation")
async def receive_dictation(
    session_id: str,
    text: StreamDeckText,
    service: StreamDeckServiceDep,
    registry: RuntimeSessionRegistryDep,
    x_streamdeck_connection: Annotated[str, Header()],
) -> StreamDeckStatus:
    """Deliver a draft only; speech must never start or modify a running control."""
    target = service.require_target(session_id, x_streamdeck_connection, registry, page="inference")
    session = registry.get(session_id)
    if session is None:
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    if session.state().control is not None or not text.task.strip():
        raise APIError("Dictation requires stopped control and non-empty speech.", "dictation_unavailable", 409)
    service.target = target.model_copy(
        update={"task": text.task, "dictation": StreamDeckTranscript(id=uuid4().hex, task=text.task)}
    )
    return service.status(registry)


@router.delete("/sessions/{session_id}")
async def disconnect_streamdeck(
    session_id: str, service: StreamDeckServiceDep, registry: RuntimeSessionRegistryDep
) -> StreamDeckStatus:
    """Release USB access without changing recording or teleoperation."""
    return service.disconnect(session_id, registry)

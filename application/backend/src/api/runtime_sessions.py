"""Watch, record with and stop the runtime sessions of this API process.

Not project-scoped: the footer shows every session, because a session in one
project can hold a camera another project wants.
"""

from typing import Annotated

import anyio
from fastapi import APIRouter, Header, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import RuntimeSessionRegistryDep, StreamDeckServiceDep
from exceptions import ResourceNotFoundError, ResourceType
from runtime_session.commands import request_command
from runtime_session.registry import RuntimeSessionRegistry
from runtime_session.session import RuntimeSession
from schemas.runtime_session import AckMessage, RecordingCommand, RuntimeSessionsMessage, SessionState

router = APIRouter(prefix="/api/runtime/sessions", tags=["Runtime Sessions"])


@router.get(
    "/ws",
    tags=["WebSocket"],
    summary="Live runtime sessions (WebSocket)",
    status_code=426,
    responses={200: {"model": RuntimeSessionsMessage, "description": "Sent on connect and on every change"}},
)
async def runtime_sessions_websocket_openapi() -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtime_sessions_websocket(websocket: WebSocket, registry: RuntimeSessionRegistryDep) -> None:  # noqa: D103 - documented on the GET stub
    await websocket.accept()
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_send_on_change, websocket, registry)
            tasks.start_soon(_until_closed, websocket)
    except* WebSocketDisconnect:
        pass


async def _until_closed(websocket: WebSocket) -> None:
    """The client sends nothing; receiving is how a closed socket is noticed while idle."""
    while True:
        await websocket.receive_text()


async def _send_on_change(websocket: WebSocket, registry: RuntimeSessionRegistry) -> None:
    with registry.watch() as changes:
        while True:
            message = RuntimeSessionsMessage(sessions=[session.info() for session in registry.sessions()])
            await websocket.send_json(message.model_dump(mode="json"))
            await changes.get()


@router.get("")
async def list_runtime_sessions(registry: RuntimeSessionRegistryDep) -> RuntimeSessionsMessage:
    """List live sessions without opening or taking ownership of one."""
    return RuntimeSessionsMessage(sessions=[session.info() for session in registry.sessions()])


def _get_session(session_id: str, registry: RuntimeSessionRegistry) -> RuntimeSession:
    session = registry.get(session_id)
    if session is None or session.closed.is_set():
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    return session


@router.get("/{session_id}/state")
async def runtime_session_state(session_id: str, registry: RuntimeSessionRegistryDep) -> SessionState:
    """Read an existing session without streaming camera frames or keeping it alive."""
    return _get_session(session_id, registry).state()


@router.post("/{session_id}/recording")
async def record_runtime_session(
    session_id: str,
    command: RecordingCommand,
    registry: RuntimeSessionRegistryDep,
    service: StreamDeckServiceDep,
    x_streamdeck_connection: Annotated[str | None, Header()] = None,
) -> AckMessage:
    """Start, save or discard an episode using the same handlers as the Studio workspace."""
    session = _get_session(session_id, registry)
    if x_streamdeck_connection is not None:
        service.require_target(session_id, x_streamdeck_connection, registry, page="record")
    return await request_command(session.bus, command.model_dump_json())


@router.post("/{session_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_runtime_session(session_id: str, registry: RuntimeSessionRegistryDep) -> None:
    """Close a session, releasing its devices. Its websocket gets a final ``session_closed`` error."""
    session = registry.get(session_id)
    if session is None:
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    await session.stop()

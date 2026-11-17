"""Watch and stop the runtime sessions of this API process.

Not project-scoped: the footer shows every session, because a session in one
project can hold a camera another project wants.
"""

import anyio
from fastapi import APIRouter, WebSocket, status
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect

from api.dependencies import RuntimeSessionRegistryDep
from exceptions import ResourceNotFoundError, ResourceType
from runtime_session.registry import RuntimeSessionRegistry
from schemas.runtime_session import RuntimeSessionsMessage

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


@router.post("/{session_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_runtime_session(session_id: str, registry: RuntimeSessionRegistryDep) -> None:
    """Close a session, releasing its devices. Its websocket gets a final ``session_closed`` error."""
    session = registry.get(session_id)
    if session is None:
        raise ResourceNotFoundError(ResourceType.RUNTIME_SESSION, session_id)
    await session.stop()

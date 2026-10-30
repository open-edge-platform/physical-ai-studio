"""The runtime session websocket: one session for the lifetime of the socket.

The socket only forwards. Every JSON command goes onto the session's bus as a
request on ``rpc/<command>``; whichever part serves it (the session, or a
worker such as the dataset worker) answers, and the answer becomes the ack.
What the session publishes on ``evt/session`` (state, errors) and every
part's status on ``evt/status/**`` go back out,
and the store is sampled at ``hz`` into one binary observation per tick. See
``schemas/runtime_session.py`` for the messages.
"""

import asyncio
from typing import Annotated, Any
from uuid import UUID

import anyio
from fastapi import APIRouter, Query, WebSocket
from fastapi.responses import Response
from fastapi.websockets import WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from api.dependencies import (
    CameraClaimRegistryDep,
    DatasetServiceDep,
    ModelServiceDep,
    ProjectCameraServiceDep,
    ProjectServiceDep,
    RobotClientFactoryDep,
    RobotServiceDep,
    RuntimeSessionRegistryDep,
)
from runtime_session import session as runtime_session
from runtime_session.bus import STATUSES, Bus, BusRequestError, NoResponderError, Subscription, command_key
from runtime_session.resolve import project_resolvers
from runtime_session.session import RuntimeSession
from runtime_session.store import ACTION, STATE, Sample
from schemas.runtime_session import (
    AckMessage,
    ObservationFrame,
    ObservationMessage,
    RuntimeSessionCommand,
    RuntimeSessionServerMessage,
    SessionError,
    SessionEvent,
    StateMessage,
    StatusMessage,
)
from workers.rate import Rate

router = APIRouter(prefix="/api/projects/{project_id}/runtime", tags=["Runtime Sessions"])

_COMMAND_TIMEOUT_S = 600.0
"""A backstop only: every handler bounds its own work (a dataset copy, a video encode)."""


class _CommandHeader(BaseModel):
    """All the socket reads of a command; whoever serves it validates the rest."""

    command: str = Field(pattern=r"^[a-z_]+$")
    request_id: str


class _SessionEnded(Exception):
    pass


@router.get(
    "/ws",
    tags=["WebSocket"],
    summary="Runtime session (WebSocket)",
    status_code=426,
    responses={
        200: {"model": RuntimeSessionServerMessage, "description": "JSON messages the server sends"},
        299: {"model": RuntimeSessionCommand, "description": "JSON commands the client sends"},
    },
)
async def runtime_session_websocket_openapi(
    project_id: UUID,  # noqa: ARG001
    hz: Annotated[float, Query(gt=0, le=60, description="Observation and frame rate")] = 20,  # noqa: ARG001
) -> Response:
    """This endpoint requires a WebSocket connection. Use `wss://` to connect."""
    return Response(status_code=426)


@router.websocket("/ws")
async def runtime_session_websocket(  # noqa: PLR0913, D103 - FastAPI dependencies; documented on the GET stub
    websocket: WebSocket,
    project_id: UUID,
    project_service: ProjectServiceDep,
    robot_service: RobotServiceDep,
    camera_service: ProjectCameraServiceDep,
    dataset_service: DatasetServiceDep,
    model_service: ModelServiceDep,
    robot_client_factory: RobotClientFactoryDep,
    registry: RuntimeSessionRegistryDep,
    camera_claims: CameraClaimRegistryDep,
    hz: Annotated[float, Query(gt=0, le=60)] = 20,
) -> None:
    await websocket.accept()
    project = await project_service.get_project_by_id(project_id)
    session = RuntimeSession(
        project_id=project_id,
        project_name=project.name,
        registry=registry,
        camera_claims=camera_claims,
        robot_factory=robot_client_factory,
        resolve=project_resolvers(project_id, robot_service, camera_service, dataset_service, model_service),
    )
    async with session:
        try:
            with (
                session.bus.subscribe(runtime_session.EVENTS, SessionEvent) as events,
                session.bus.subscribe(STATUSES, StatusMessage) as statuses,
            ):
                await websocket.send_json(StateMessage(data=session.state()).model_dump(mode="json"))
                # anyio, not asyncio.TaskGroup: Starlette cancels through anyio scopes, which asyncio's group can leak.
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(_forward_commands, websocket, session.bus)
                    tasks.start_soon(_forward_events, websocket, events)
                    tasks.start_soon(_forward_events, websocket, statuses)
                    tasks.start_soon(_sample_store, websocket, session, hz)
        except* (WebSocketDisconnect, _SessionEnded):
            pass
        except* Exception as errors:
            logger.opt(exception=errors).error("Runtime session {} websocket failed", session.id)


async def _forward_commands(websocket: WebSocket, bus: Bus) -> None:
    """Put each command onto the bus, one at a time and in order, and ack it with the reply."""
    while True:
        raw = await websocket.receive_text()
        ack = await _request(bus, raw)
        await websocket.send_json(ack.model_dump(mode="json"))


async def _request(bus: Bus, raw: str) -> AckMessage:
    try:
        header = _CommandHeader.model_validate_json(raw)
    except ValidationError as error:
        return AckMessage(request_id="", ok=False, error=SessionError(message=str(error), error_code="invalid_command"))

    def failed(message: str, error_code: str) -> AckMessage:
        return AckMessage(
            request_id=header.request_id, ok=False, error=SessionError(message=message, error_code=error_code)
        )

    try:
        result: dict[str, Any] = await bus.request(
            command_key(header.command), raw, dict[str, Any], timeout=_COMMAND_TIMEOUT_S
        )
    except BusRequestError as error:
        # The handler's details (``parts`` for ``incompatible_parts``, say) go to the client as they are.
        return AckMessage(
            request_id=header.request_id,
            ok=False,
            error=SessionError(message=error.message, error_code=error.error_code, **error.data),
        )
    except NoResponderError:
        # Nothing serves it now: an unknown command, or one for a part that is not loaded (a dataset, say).
        return failed(f"Nothing can {header.command} right now; load what it needs first.", "command_unavailable")
    except TimeoutError:
        return failed(f"{header.command} did not finish in time", "command_timeout")
    return AckMessage(request_id=header.request_id, ok=True, result=result or None)


async def _forward_events(websocket: WebSocket, events: Subscription[Any]) -> None:
    while True:
        event = await events.next()
        await websocket.send_json(event.model_dump(mode="json"))
        if event.type == "error" and not event.session_alive:
            await websocket.close()
            raise _SessionEnded


def _named(sample: Sample | None) -> dict[str, float] | None:
    return None if sample is None else dict(zip(sample.names, map(float, sample.value), strict=True))


async def _sample_store(websocket: WebSocket, session: RuntimeSession, hz: float) -> None:
    """Send the store at ``hz``: one observation per tick with the joints and every new camera frame."""
    from utils.jpeg import encode_jpeg_rgb

    rate = Rate(hz, name=f"session-sampler-{session.id}")
    sent: dict[str, float] = {}
    while True:
        store = session.store
        if store is None:
            sent.clear()
        else:
            state, action = store.read(STATE), store.read(ACTION)
            joints_at = max((sample.timestamp for sample in (state, action) if sample is not None), default=None)
            frames = {
                key: frame
                for key in store.spec.observations
                if (frame := store.read_observation(key)) is not None and frame.timestamp != sent.get(key)
            }
            if frames or (joints_at is not None and joints_at != sent.get(STATE)):
                jpegs = await asyncio.gather(
                    *(asyncio.to_thread(encode_jpeg_rgb, frame.value) for frame in frames.values())
                )
                header = ObservationMessage(
                    timestamp=max([joints_at or 0.0, *(frame.timestamp for frame in frames.values())]),
                    state=_named(state),
                    action=_named(action),
                    frames=[
                        ObservationFrame(key=key, timestamp=frame.timestamp, size=len(jpeg))
                        for (key, frame), jpeg in zip(frames.items(), jpegs, strict=True)
                    ],
                )
                await websocket.send_bytes(observation_message(header, jpegs))
                if joints_at is not None:
                    sent[STATE] = joints_at
                sent.update((key, frame.timestamp) for key, frame in frames.items())
        await rate.sleep()


def observation_message(header: ObservationMessage, jpegs: list[bytes]) -> bytes:
    """``[u32 big-endian header length][header JSON][jpeg bytes, in the order of header.frames]``."""
    encoded = header.model_dump_json().encode()
    return len(encoded).to_bytes(4, "big") + encoded + b"".join(jpegs)

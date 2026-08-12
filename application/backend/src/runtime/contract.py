from __future__ import annotations

import queue
import threading
from collections import deque
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol
from uuid import UUID  # noqa: TC003 — pydantic resolves annotations from module globals at build time

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from schemas import InferenceDevice

if TYPE_CHECKING:
    from collections.abc import Iterable

FollowerSource = Literal["hold", "teleop", "policy"]


class CommandBase(BaseModel):
    request_id: str | None = None


class SetFollowerSourceCommand(CommandBase):
    command: Literal["set_follower_source"] = "set_follower_source"
    follower_source: FollowerSource


class DisconnectCommand(CommandBase):
    command: Literal["disconnect"] = "disconnect"


class LoadModelCommand(CommandBase):
    command: Literal["load_model"] = "load_model"
    # An identifier, never a row: the API resolves it against the project the
    # way the websocket handshake already resolves follower_id and leader_id,
    # so a client cannot name the directory the backend loads a model from.
    #
    # The session owns no database, so it derives the rest from the filesystem:
    # the artifacts live at ``models_dir / <model_id>`` (both writers keep that
    # invariant, see training_worker and model_import_service) and the policy
    # name comes from the export manifest, not from the model row.
    model_id: UUID
    inference_device: InferenceDevice


class LoadDatasetCommand(CommandBase):
    command: Literal["load_dataset"] = "load_dataset"
    # ``datasets_dir / <dataset_id>`` is already the only definition of a
    # dataset's location; see the Dataset schema.
    dataset_id: UUID


class StartTaskCommand(CommandBase):
    command: Literal["start_task"] = "start_task"
    task: str


class StopTaskCommand(CommandBase):
    command: Literal["stop_task"] = "stop_task"


class StartRecordingCommand(CommandBase):
    command: Literal["start_recording"] = "start_recording"
    task: str


class SaveEpisodeCommand(BaseModel):
    command: Literal["save_episode"] = "save_episode"
    request_id: str


class DiscardEpisodeCommand(BaseModel):
    command: Literal["discard_episode"] = "discard_episode"
    request_id: str


Command = Annotated[
    SetFollowerSourceCommand
    | DisconnectCommand
    | LoadModelCommand
    | LoadDatasetCommand
    | StartTaskCommand
    | StopTaskCommand
    | StartRecordingCommand
    | SaveEpisodeCommand
    | DiscardEpisodeCommand,
    Field(discriminator="command"),
]
CommandAdapter: TypeAdapter[Command] = TypeAdapter(Command)


class ObservationEvent(BaseModel):
    event: Literal["observation"] = "observation"
    data: dict[str, float]


class StateData(BaseModel):
    model_config = ConfigDict(extra="allow")

    connected: bool
    follower_source: FollowerSource


class StateEvent(BaseModel):
    event: Literal["state"] = "state"
    data: StateData


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str
    error_code: str


class LifecycleData(BaseModel):
    event: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LifecycleEvent(BaseModel):
    event: Literal["lifecycle"] = "lifecycle"
    data: LifecycleData


class AckData(BaseModel):
    request_id: str
    ok: bool
    error: str | None = None


class AckEvent(BaseModel):
    event: Literal["ack"] = "ack"
    data: AckData


RuntimeEvent = ObservationEvent | StateEvent | ErrorEvent | LifecycleEvent | AckEvent


class CommandMailbox(Protocol):
    def apply(self, command: Command) -> None:
        """Store a command for the runtime control thread."""

    def drain(self) -> Iterable[Command]:
        """Return pending commands in arrival order."""


class EventSink(Protocol):
    def emit(self, event: RuntimeEvent) -> None:
        """Publish one runtime event."""


class InMemoryCommandMailbox:
    """Thread-safe, in-process command binding used by the phase A host."""

    def __init__(self) -> None:
        self._commands: queue.SimpleQueue[Command] = queue.SimpleQueue()

    def apply(self, command: Command) -> None:
        self._commands.put(command)

    def drain(self) -> Iterable[Command]:
        while True:
            try:
                yield self._commands.get_nowait()
            except queue.Empty:
                return


class QueueEventSink:
    """Thread-safe event sink that keeps only the newest observation."""

    def __init__(self) -> None:
        self._events: deque[RuntimeEvent] = deque()
        self._lock = threading.Lock()

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            if isinstance(event, ObservationEvent):
                self._events = deque(item for item in self._events if not isinstance(item, ObservationEvent))
            self._events.append(event)

    def get_nowait(self) -> RuntimeEvent:
        with self._lock:
            if not self._events:
                raise queue.Empty
            return self._events.popleft()

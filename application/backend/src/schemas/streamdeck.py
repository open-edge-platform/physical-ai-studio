"""The host helper's lease and the workspace it has explicitly been attached to."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from schemas.runtime_session import ModelControlConfig

DictationStatus = Literal["idle", "preparing", "listening", "transcribing", "done", "error"]


class StreamDeckText(BaseModel):
    task: str = Field(max_length=20_000)


class StreamDeckTask(StreamDeckText):
    page: Literal["record", "inference"] = "record"
    inference: ModelControlConfig = Field(default_factory=ModelControlConfig)


class StreamDeckTranscript(StreamDeckText):
    id: str


class StreamDeckInferenceCommand(BaseModel):
    action: Literal["start", "stop"]
    request_id: str


class StreamDeckTarget(StreamDeckTask):
    session_id: str
    connection_id: str
    dictation: StreamDeckTranscript | None = None


class StreamDeckHeartbeat(BaseModel):
    helper_id: UUID
    connection_id: str | None = None
    device: str | None = Field(default=None, max_length=200)
    error: str | None = Field(default=None, max_length=2000)
    dictation_status: DictationStatus = "idle"


class StreamDeckStatus(BaseModel):
    helper_online: bool
    can_start_native: bool = False
    helper_starting: bool = False
    connected: bool
    device: str | None
    error: str | None
    target: StreamDeckTarget | None
    dictation_status: DictationStatus = "idle"

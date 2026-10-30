"""Runtime session websocket messages and the session listing.

Client to server: one JSON command per message, each with a ``request_id``
that its ``ack`` echoes. Server to client: ``ack``, ``state`` on change and
``error`` as JSON text, and ``observation`` at the sampling rate as one binary
message per tick: ``[u32 big-endian header length][ObservationMessage JSON][jpeg bytes]``,
the JPEGs concatenated in the order of ``frames``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID  # noqa: TC003 - pydantic resolves annotations at runtime

from pydantic import BaseModel, Field

from schemas.base import UTCDatetime
from schemas.environment import CameraEnvironmentConfiguration, RobotEnvironmentConfiguration


class EnvironmentDraft(BaseModel):
    """An environment to load, saved or still being edited in the form."""

    id: UUID | None = None
    """The saved environment this is, if it is one; the session reports it back."""
    name: str = "Unsaved environment"
    robots: list[RobotEnvironmentConfiguration]
    cameras: list[CameraEnvironmentConfiguration] = Field(default_factory=list)


class TeleopControlConfig(BaseModel):
    """Drive the follower from the environment's leader arm."""

    type: Literal["teleop"] = "teleop"
    hz: float = Field(100, gt=0, le=500, description="How often the leader is read; each reading is one command")
    grace_s: float = Field(
        3,
        ge=0,
        le=30,
        description="On every start, the follower closes the gap to the leader over this long instead of jumping",
    )
    slowdown: float | None = Field(
        None,
        ge=1,
        le=10,
        description="Relative mode: the follower moves by the leader's movement since the start, divided by this",
    )


class MoveControlConfig(BaseModel):
    """Move the follower from where it is to ``position`` over ``duration_s``, then hold there."""

    type: Literal["move"] = "move"
    position: dict[str, float] = Field(description="Target by joint name (`<joint>.pos`); joints left out stay put")
    duration_s: float = Field(gt=0, le=60)
    hz: float = Field(50, gt=0, le=500)


class ReplayControlConfig(BaseModel):
    """Play one of the loaded dataset's episodes on the follower, at the dataset's fps, then hold its last frame.

    Only episodes on disk: those recorded this session reach the dataset when the recording is finished.
    """

    type: Literal["replay"] = "replay"
    episode_index: int = Field(ge=0)
    grace_s: float = Field(3, ge=0, le=30, description="The follower moves to the episode's first frame over this long")


class ExecutionConfig(BaseModel):
    """How a model's action chunks are requested, as ``physicalai.runtime`` runs them."""

    mode: Literal["sync", "async"] = "sync"
    """``sync`` infers in the control's tick and waits; ``async`` infers alongside, while the queue plays."""
    request_threshold: float = Field(
        default=0.5, ge=0, le=1, description="Request the next chunk once this fraction of the current one is left"
    )


class ModelControlConfig(BaseModel):
    """Drive the follower with the loaded model."""

    type: Literal["model"] = "model"
    task: str = ""
    """What the model is asked to do, for a policy that takes a task."""
    hz: float = Field(30, gt=0, le=500, description="How often an action is taken from the model's queue")
    execution: ExecutionConfig = ExecutionConfig()
    grace_s: float = Field(
        3,
        ge=0,
        le=30,
        description="On every start, the follower closes the gap to the model's actions over this long",
    )


ControlConfig = Annotated[
    TeleopControlConfig | MoveControlConfig | ReplayControlConfig | ModelControlConfig, Field(discriminator="type")
]


class LoadEnvironmentCommand(BaseModel):
    command: Literal["load_environment"] = "load_environment"
    request_id: str
    environment: EnvironmentDraft
    unload_incompatible: bool = False
    """Replace a loaded environment even if a loaded part no longer fits it; that part is unloaded.

    Without it such a load fails with ``incompatible_parts`` and changes nothing.
    """


class UnloadEnvironmentCommand(BaseModel):
    command: Literal["unload_environment"] = "unload_environment"
    request_id: str


class ReloadEnvironmentCommand(BaseModel):
    """Connect the loaded environment's devices again, after one failed; the store and dataset stay."""

    command: Literal["reload_environment"] = "reload_environment"
    request_id: str


class ReloadEnvironment(BaseModel):
    """What the session asks the environment, on ``rpc/environment/reload``."""


class LoadModelCommand(BaseModel):
    """Load one of a model's exports to run. Needs an environment whose joints and cameras fit it."""

    command: Literal["load_model"] = "load_model"
    request_id: str
    model_id: UUID
    backend: str
    """One of the model's ``available_backends``."""
    device: str = "auto"


class UnloadModelCommand(BaseModel):
    command: Literal["unload_model"] = "unload_model"
    request_id: str


class LoadDatasetCommand(BaseModel):
    """Record into this dataset. Needs an environment whose joints and cameras fit it."""

    command: Literal["load_dataset"] = "load_dataset"
    request_id: str
    dataset_id: UUID
    fps: int | None = Field(
        None, ge=1, le=100, description="The rate to record a dataset with nothing recorded at; others keep their own"
    )


class UnloadDatasetCommand(BaseModel):
    """Stop recording into the dataset and copy what was recorded back into it."""

    command: Literal["unload_dataset"] = "unload_dataset"
    request_id: str


class StartEpisodeCommand(BaseModel):
    command: Literal["start_episode"] = "start_episode"
    request_id: str
    task: str


class SaveEpisodeCommand(BaseModel):
    command: Literal["save_episode"] = "save_episode"
    request_id: str


class DiscardEpisodeCommand(BaseModel):
    command: Literal["discard_episode"] = "discard_episode"
    request_id: str


class SetTeleopSlowdownCommand(BaseModel):
    """While teleoperating, switch to relative teleop slowed down by ``slowdown``, or back to absolute with ``null``.

    The control keeps running, so the follower is driven throughout the switch.
    """

    command: Literal["set_teleop_slowdown"] = "set_teleop_slowdown"
    request_id: str
    slowdown: float | None = Field(None, ge=1, le=10)


class SetControlCommand(BaseModel):
    """Switch what drives the follower. ``null`` holds it where it is."""

    command: Literal["set_control"] = "set_control"
    request_id: str
    control: ControlConfig | None


RuntimeSessionCommand = Annotated[
    LoadEnvironmentCommand
    | UnloadEnvironmentCommand
    | ReloadEnvironmentCommand
    | SetControlCommand
    | SetTeleopSlowdownCommand
    | LoadDatasetCommand
    | UnloadDatasetCommand
    | LoadModelCommand
    | UnloadModelCommand
    | StartEpisodeCommand
    | SaveEpisodeCommand
    | DiscardEpisodeCommand,
    Field(discriminator="command"),
]


class IncompatiblePart(BaseModel):
    part: Literal["dataset", "model"]
    name: str
    reasons: list[str]


class SessionError(BaseModel):
    message: str
    error_code: str
    parts: list[IncompatiblePart] | None = None
    """With ``incompatible_parts``: what the command would unload, and why."""


class AckMessage(BaseModel):
    type: Literal["ack"] = "ack"
    request_id: str
    ok: bool
    error: SessionError | None = None
    result: dict[str, Any] | None = None


class SessionCamera(BaseModel):
    camera_id: UUID
    name: str
    key: str
    """The store and frame key; also the key a dataset records this camera under."""
    height: int
    width: int


class SessionEnvironment(BaseModel):
    environment_id: UUID | None
    """The saved environment, or ``None`` for an unsaved one from the form."""
    name: str
    robot_id: UUID
    leader_id: UUID | None
    joint_names: list[str]
    cameras: list[SessionCamera]


class SessionDataset(BaseModel):
    dataset_id: UUID
    name: str
    fps: int
    fps_fixed: bool
    """Whether the dataset already has episodes, and so its own rate; a new one takes the rate it is loaded with."""
    is_recording: bool
    episodes_recorded: int
    """Saved in this session and not yet copied into the dataset, which happens when it is unloaded."""


class SessionModel(BaseModel):
    model_id: UUID
    name: str
    backend: str
    device: str


class SessionState(BaseModel):
    session_id: str
    environment: SessionEnvironment | None = None
    control: ControlConfig | None = None
    """What drives the follower; ``None`` holds it still."""
    dataset: SessionDataset | None = None
    model: SessionModel | None = None


class StateMessage(BaseModel):
    type: Literal["state"] = "state"
    data: SessionState


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    component: str
    message: str
    error_code: str
    session_alive: bool
    """False when the session has ended; the socket closes after this message."""


ComponentState = Literal["starting", "running", "failed", "stopped"]


class StatusMessage(BaseModel):
    """A part of the session or one of its devices saying how it is, on ``evt/status/<component>``.

    ``component`` is a path: ``environment``, ``control`` and ``dataset`` are the
    session's parts, ``environment/follower``, ``environment/leader`` and
    ``environment/<camera key>`` the environment's devices. Devices report only
    ``starting``, ``running`` and ``failed``: a part that starts or stops takes its
    devices with it, so their earlier statuses no longer apply.
    """

    type: Literal["status"] = "status"
    component: str
    state: ComponentState
    error: str | None = None
    """Why it failed."""


SessionEvent = Annotated[StateMessage | ErrorMessage, Field(discriminator="type")]
"""What the session publishes on ``evt/session`` for its socket to forward."""


class CommandDone(BaseModel):
    """The reply to a command that has no result."""


class ObservationFrame(BaseModel):
    key: str
    timestamp: float
    size: int
    """Bytes of this JPEG in the message body."""


class ObservationMessage(BaseModel):
    """The header of a binary observation message; ``frames`` lists only frames new since the last one."""

    type: Literal["observation"] = "observation"
    timestamp: float
    state: dict[str, float] | None
    action: dict[str, float] | None
    frames: list[ObservationFrame]


RuntimeSessionServerMessage = Annotated[
    AckMessage | StateMessage | ErrorMessage | StatusMessage | ObservationMessage,
    Field(discriminator="type"),
]


class RuntimeSessionInfo(BaseModel):
    """One live runtime session, for the footer and the robot list."""

    session_id: str
    project_id: UUID
    project_name: str
    started_at: UTCDatetime
    environment_name: str | None
    robot_ids: list[UUID]
    camera_ids: list[UUID]


class RuntimeSessionsMessage(BaseModel):
    """Sent by ``/api/runtime/sessions/ws`` on connect and whenever the list changes."""

    sessions: list[RuntimeSessionInfo]

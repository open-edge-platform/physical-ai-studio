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


class AutopilotControlConfig(BaseModel):
    """Follow the simulation's demonstrator while autopilot recording runs.

    Only the session starts it, with ``start_autopilot_recording``; ``set_control`` refuses it.
    """

    type: Literal["autopilot"] = "autopilot"
    hz: float = Field(100, gt=0, le=500, description="How often the demonstrator's targets are read")


ControlConfig = Annotated[
    TeleopControlConfig | ModelControlConfig | AutopilotControlConfig, Field(discriminator="type")
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


EXPECTED_DATASET = Field(
    None,
    description=(
        "If set, the command is refused unless this dataset is the one recording. "
        "Automatic recording sets it, since another dataset may be loaded before its command runs"
    ),
)


class StartEpisodeCommand(BaseModel):
    command: Literal["start_episode"] = "start_episode"
    request_id: str
    task: str
    dataset_id: UUID | None = EXPECTED_DATASET
    episode_id: str | None = Field(
        None, max_length=64, description="The client's name for the episode, for ending exactly this one"
    )


EXPECTED_EPISODE = Field(
    None,
    max_length=64,
    description="Refuse unless the open episode is the one started with this ``episode_id``",
)


class SaveEpisodeCommand(BaseModel):
    command: Literal["save_episode"] = "save_episode"
    request_id: str
    dataset_id: UUID | None = EXPECTED_DATASET
    episode_id: str | None = EXPECTED_EPISODE


class DiscardEpisodeCommand(BaseModel):
    command: Literal["discard_episode"] = "discard_episode"
    request_id: str
    dataset_id: UUID | None = EXPECTED_DATASET
    episode_id: str | None = EXPECTED_EPISODE


class AutopilotRecordingOptions(BaseModel):
    """How autopilot recording runs. ``None`` keeps the simulation's default for its scene."""

    max_episodes: Annotated[int, Field(ge=0, le=10_000, description="Stop after this many saved episodes; 0 never")] = 0
    max_attempts: Annotated[int | None, Field(ge=1, le=5, description="The demonstrator's attempts per episode")] = None
    recovery_fraction: Annotated[
        float | None,
        Field(ge=0.0, le=0.5, allow_inf_nan=False, description="Share of episodes that start from a stuck pose"),
    ] = None


class StartAutopilotRecordingCommand(BaseModel):
    """Record the simulation's demonstrator into the loaded dataset until stopped or the budget is used up.

    The session leads the follower with the demonstrator, starts each episode with the episode's
    task, and keeps successes only. Manual episode commands are refused meanwhile.
    """

    command: Literal["start_autopilot_recording"] = "start_autopilot_recording"
    request_id: str
    dataset_id: UUID = Field(..., description="The loaded dataset to record into; refused if another is loaded")
    options: AutopilotRecordingOptions = Field(default_factory=lambda: AutopilotRecordingOptions())  # noqa: PLW0108


class StopAutopilotRecordingCommand(BaseModel):
    """Stop autopilot recording: the follower holds, and an episode being recorded is discarded."""

    command: Literal["stop_autopilot_recording"] = "stop_autopilot_recording"
    request_id: str


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
    | LoadDatasetCommand
    | UnloadDatasetCommand
    | LoadModelCommand
    | UnloadModelCommand
    | StartEpisodeCommand
    | SaveEpisodeCommand
    | DiscardEpisodeCommand
    | StartAutopilotRecordingCommand
    | StopAutopilotRecordingCommand,
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
    autopilot: bool = False
    """Whether the simulation's autopilot leads the follower: it leads only while autopilot recording runs."""
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
    tasks: list[str] = []
    """The tasks the model takes, when it only knows a fixed set; empty if it takes any text."""


AutopilotRecordingPhase = Literal[
    "starting", "preparing", "recording", "saving", "stopping", "stopped", "done", "failed"
]
"""``starting``; ``preparing`` an episode (the simulation lays out its scene, then every camera must be fresh);
``recording`` it; ``saving`` or discarding it; ``stopping``; then ``done`` (the budget is used up), ``stopped``
(by the user, or by something else the session did) or ``failed``."""

DiscardReason = Literal["failure", "stalled", "aborted", "stale_stream", "too_short", "dataset", "save_failed"]
"""Why an episode was not kept: the demonstrator failed or stalled; the episode was aborted (by a stop, the
simulation, or a leader sample of another episode); a camera went stale; too few frames; the dataset discarded
it (a camera was lost); saving it failed. A success after a retry is kept."""


class AutopilotEpisodeSummary(BaseModel):
    """How the last episode of autopilot recording ended."""

    id: str
    """The simulation's id of the episode."""
    task: str | None = None
    outcome: str | None = None
    """The simulation's outcome: ``success``, ``failure``, ``stalled`` or ``aborted``."""
    clean: bool | None = None
    kept: bool
    reason: str | None = None
    """Why it was discarded, or the simulation's note on a kept one (a regrasp, say)."""
    discard_reason: DiscardReason | None = None
    frames: int | None = None


class AutopilotRecordingState(BaseModel):
    """Autopilot recording in this session: what it does now, and how many episodes it kept and discarded."""

    phase: AutopilotRecordingPhase
    message: str = ""
    scene: str | None = None
    task: str | None = None
    """The task of the episode being prepared or recorded."""
    saved: int = 0
    discarded: int = 0
    discards: dict[DiscardReason, int] = Field(default_factory=dict, description="Discarded episodes by reason")
    max_episodes: int = 0
    last: AutopilotEpisodeSummary | None = None


class SessionState(BaseModel):
    session_id: str
    rerun_url: str | None = None
    """Hosted web viewer for the session's stream; local streams require a browser on the backend host."""
    environment: SessionEnvironment | None = None
    control: ControlConfig | None = None
    """What drives the follower; ``None`` holds it still."""
    dataset: SessionDataset | None = None
    model: SessionModel | None = None
    autopilot_recording: AutopilotRecordingState | None = None
    """The running or last autopilot recording of this session; ``None`` before the first."""


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

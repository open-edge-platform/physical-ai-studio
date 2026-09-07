# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read models for live runtime sessions.

Every field but ``session_name`` and ``status`` is optional on purpose. A session
publishes its metadata from a separate process, so the API treats that payload as
untrusted input: a session still starting has published almost none of it, one that
stopped answering has published none of it at all, and the fields it does publish
can be the wrong shape.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — pydantic resolves annotations at build time
from enum import StrEnum
from uuid import UUID  # noqa: TC003 — pydantic resolves annotations at build time

from pydantic import BaseModel, Field

from runtime.contract import FollowerSource


class RuntimeSessionStatus(StrEnum):
    """Where a session is in its life, as far as the API can tell."""

    STARTING = "starting"
    """Metadata answers, but the hardware is not connected yet."""

    RUNNING = "running"
    """The last state event reported a connected robot."""

    STOPPED = "stopped"
    """A shutdown lifecycle event was published; the process is on its way out."""

    ERROR = "error"
    """The session published a fatal event."""

    UNREACHABLE = "unreachable"
    """Holds a live lock, but its metadata did not answer or made no sense.

    Not hidden from the list: a session that holds a robot and will not talk is
    the one most likely to need stopping.
    """


class RuntimeSessionActivity(BaseModel):
    """What a session is *doing*, as opposed to where it is in its life.

    Mirrors ``runtime.contract.StateData``. Absent until the session has
    published a connected state event.
    """

    connected: bool
    follower_source: FollowerSource
    model_loaded: bool | None = None
    task: str | None = None
    dataset_loaded: bool | None = None
    is_recording: bool | None = None
    episodes_recorded: int | None = None


class RuntimeSessionError(BaseModel):
    """The last fatal event a session published."""

    message: str
    error_code: str


class RuntimeSessionInfo(BaseModel):
    """One live runtime session."""

    session_name: str
    """``rt-<follower uuid>``. The handle the stop endpoint takes."""

    follower_id: UUID | None = None
    """Parsed out of ``session_name``. ``None`` when a stale lock does not parse."""

    status: RuntimeSessionStatus
    pid: int | None = None

    follower_name: str | None = None
    leader_name: str | None = None

    started_at: datetime | None = None
    idle_timeout_s: float | None = None

    attached: bool | None = None
    """Whether any client is subscribed. ``False`` means nobody is watching this arm."""

    idle_deadline: datetime | None = None
    """When an unattached session shuts itself down. Set only while ``attached`` is False."""

    camera_keys: list[str] = Field(default_factory=list)
    activity: RuntimeSessionActivity | None = None
    error: RuntimeSessionError | None = None


class RuntimeSessionCount(BaseModel):
    """How many sessions hold a live lock.

    Answered from the lock directory alone, so the footer can poll it without
    opening a transport session per runtime session.
    """

    count: int

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Read models for the runtime sessions this API process is running."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 — pydantic resolves annotations at build time
from enum import StrEnum
from uuid import UUID  # noqa: TC003 — pydantic resolves annotations at build time

from pydantic import BaseModel, Field

from runtime.contract import FollowerSource


class RuntimeSessionStatus(StrEnum):
    """Where a session is in its life, as far as the API can tell."""

    STARTING = "starting"
    """The worker is running, but the hardware is not connected yet."""

    RUNNING = "running"
    """The last state event reported a connected robot."""

    STOPPED = "stopped"
    """A stop was requested; the worker is tearing down."""

    ERROR = "error"
    """The session reported a fatal error."""


class RuntimeSessionActivity(BaseModel):
    """What a session is *doing*, as opposed to where it is in its life.

    Mirrors ``runtime.contract.StateData``. Absent until the session has
    reported its first state.
    """

    connected: bool
    follower_source: FollowerSource
    model_loaded: bool | None = None
    task: str | None = None
    dataset_loaded: bool | None = None
    is_recording: bool | None = None
    episodes_recorded: int | None = None


class RuntimeSessionError(BaseModel):
    """The fatal error a session reported."""

    message: str
    error_code: str


class RuntimeSessionInfo(BaseModel):
    """One live runtime session."""

    session_name: str
    """``rt-<follower uuid>``. The handle the stop endpoint takes."""

    follower_id: UUID

    status: RuntimeSessionStatus
    pid: int | None = None

    follower_name: str | None = None
    leader_name: str | None = None

    started_at: datetime

    camera_keys: list[str] = Field(default_factory=list)
    activity: RuntimeSessionActivity | None = None
    error: RuntimeSessionError | None = None


class RuntimeSessionCount(BaseModel):
    """How many runtime sessions are running."""

    count: int

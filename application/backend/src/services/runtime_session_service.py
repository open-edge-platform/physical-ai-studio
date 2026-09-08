# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discover, describe and stop the runtime sessions running on this host.

Runtime sessions are detached child processes started with
``start_new_session=True``, so they outlive the API process that spawned them and
nothing in this process holds a handle to one. Everything here therefore works
from two host-local facts instead: the on-disk lock directory, and each session's
metadata queryable. That is what makes a session started before the last restart
reachable at all.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from runtime.owner import probe_session_metadata, stop_runtime_session
from runtime.transport.lock import live_session_pid, registered_session_names
from schemas.runtime_session import (
    RuntimeSessionActivity,
    RuntimeSessionError,
    RuntimeSessionInfo,
    RuntimeSessionStatus,
)


def _follower_id(session_name: str) -> UUID | None:
    """Recover the follower robot id a session name was built from."""
    try:
        return UUID(session_name.removeprefix("rt-"))
    except ValueError:
        return None


def _status(metadata: dict[str, Any]) -> RuntimeSessionStatus:
    """Read the published status, treating anything unrecognised as unreachable."""
    try:
        return RuntimeSessionStatus(metadata.get("status"))
    except ValueError:
        return RuntimeSessionStatus.UNREACHABLE


def _timestamp(value: Any) -> datetime | None:
    """Convert published epoch seconds to an aware datetime."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _model(model: type[RuntimeSessionActivity | RuntimeSessionError], payload: Any) -> Any:
    """Validate a nested payload, dropping it whole if it does not fit.

    ``StateData`` allows extra keys, so this is also what stops unknown fields
    from leaking into the response.
    """
    if not isinstance(payload, dict):
        return None
    try:
        return model.model_validate(payload)
    except ValidationError:
        return None


class RuntimeSessionService:
    """Host-wide view of the runtime sessions holding a robot."""

    def count(self) -> int:
        """Count sessions holding a live lock. Blocking, but only a directory read."""
        return len(registered_session_names())

    def describe(self, session_name: str) -> RuntimeSessionInfo:
        """Probe one session and map its metadata onto the read model. Blocking.

        A session that holds the lock but does not answer is still described,
        from the lock file alone. Hiding it would hide the case most likely to
        need a human: something is holding a robot and cannot say why.
        """
        metadata = probe_session_metadata(session_name)
        if metadata is None:
            return RuntimeSessionInfo(
                session_name=session_name,
                follower_id=_follower_id(session_name),
                status=RuntimeSessionStatus.UNREACHABLE,
                pid=live_session_pid(session_name),
            )

        pid = metadata.get("pid")
        camera_keys = metadata.get("camera_keys")
        attached = metadata.get("attached")
        # Guard the event, not the metadata: metadata is a dict by here, while
        # "state" is absent until a session publishes one and can be any shape.
        state = metadata.get("state")
        activity = state.get("data") if isinstance(state, dict) else None
        return RuntimeSessionInfo(
            session_name=session_name,
            follower_id=_follower_id(session_name),
            status=_status(metadata),
            pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else live_session_pid(session_name),
            follower_name=_text(metadata.get("follower_name")),
            leader_name=_text(metadata.get("leader_name")),
            started_at=_timestamp(metadata.get("started_at")),
            idle_timeout_s=_number(metadata.get("idle_timeout_s")),
            attached=attached if isinstance(attached, bool) else None,
            idle_deadline=_timestamp(metadata.get("idle_deadline")),
            camera_keys=[key for key in camera_keys if isinstance(key, str)] if isinstance(camera_keys, list) else [],
            activity=_model(RuntimeSessionActivity, activity),
            error=_model(RuntimeSessionError, metadata.get("error")),
        )

    async def list_sessions(self) -> list[RuntimeSessionInfo]:
        """Describe every session holding a lock.

        Probes run concurrently, so the response costs one probe timeout rather
        than one per session. Probing does not subscribe to a session, so listing
        cannot keep an abandoned one alive past its idle timeout.
        """
        names = await asyncio.to_thread(registered_session_names)
        if not names:
            return []
        return list(await asyncio.gather(*(asyncio.to_thread(self.describe, name) for name in names)))

    async def stop(self, session_name: str) -> bool:
        """Terminate a session. Returns whether it actually let go of its lock.

        ``stop_runtime_session`` suppresses its signal errors and returns
        nothing, so this confirms rather than assumes: a worker wedged in
        uninterruptible I/O on a robot's device survives even SIGKILL, and
        reporting that as success is the one outcome that would make this
        untrustworthy.

        Idempotent, so two clients racing the same stop both succeed.
        """
        await asyncio.to_thread(stop_runtime_session, session_name)
        return await asyncio.to_thread(live_session_pid, session_name) is None

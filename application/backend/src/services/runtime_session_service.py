# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Describe and stop the runtime sessions this API process is running."""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.runtime_session import (
    RuntimeSessionActivity,
    RuntimeSessionError,
    RuntimeSessionInfo,
    RuntimeSessionStatus,
)

if TYPE_CHECKING:
    from runtime.handle import RuntimeSessionHandle
    from runtime.registry import RuntimeSessionRegistry


class RuntimeSessionService:
    """View of the runtime sessions holding a robot."""

    def __init__(self, registry: RuntimeSessionRegistry) -> None:
        self._registry = registry

    def count(self) -> int:
        return self._registry.count()

    @staticmethod
    def describe(handle: RuntimeSessionHandle) -> RuntimeSessionInfo:
        state = handle.state
        error = handle.error
        return RuntimeSessionInfo(
            session_name=handle.session_name,
            follower_id=handle.follower_id,
            status=RuntimeSessionStatus(handle.status),
            pid=handle.pid,
            follower_name=handle.follower_name,
            leader_name=handle.leader_name,
            started_at=handle.started_at,
            camera_keys=list(handle.camera_keys),
            activity=None if state is None else RuntimeSessionActivity.model_validate(state.model_dump()),
            error=None if error is None else RuntimeSessionError(message=error.message, error_code=error.error_code),
        )

    async def list_sessions(self) -> list[RuntimeSessionInfo]:
        return [self.describe(handle) for handle in self._registry.list()]

    async def stop(self, session_name: str) -> bool:
        """Stop a session. Returns whether its process is gone. Idempotent."""
        return await self._registry.stop(session_name)

"""Process-local registry of the runtime sessions this API process is running."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from typing import TYPE_CHECKING

from exceptions import RuntimeSessionBusyError

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass

    from runtime.handle import RuntimeSessionHandle


class RuntimeSessionRegistry:
    """At most one session per follower robot, owned by the websocket that started it.

    In-memory on purpose: sessions are child processes of this API process and
    end with it, so there is nothing to discover across restarts.
    """

    def __init__(self, stop_event: EventClass | None = None) -> None:
        # The scheduler's application-wide stop event; every worker also watches it.
        self.stop_event: EventClass = stop_event if stop_event is not None else mp.Event()
        self._sessions: dict[str, RuntimeSessionHandle] = {}

    async def acquire(self, handle: RuntimeSessionHandle) -> None:
        """Claim the follower for ``handle``.

        A session whose websocket already closed is still tearing down (e.g. a
        page refresh); wait for it rather than reporting the robot busy.

        Raises:
            RuntimeSessionBusyError: another live websocket holds this follower.
        """
        name = handle.session_name
        while True:
            existing = self._sessions.get(name)
            if existing is None:
                self._sessions[name] = handle
                return
            if not existing.stopping and existing.is_alive():
                raise RuntimeSessionBusyError(robot_name=existing.follower_name, pid=existing.pid)
            await asyncio.to_thread(existing.stop)
            if self._sessions.get(name) is existing:
                del self._sessions[name]

    def release(self, handle: RuntimeSessionHandle) -> None:
        """Drop ``handle`` if it still holds its slot."""
        name = handle.session_name
        if self._sessions.get(name) is handle:
            del self._sessions[name]

    def get(self, session_name: str) -> RuntimeSessionHandle | None:
        return self._sessions.get(session_name)

    def list(self) -> list[RuntimeSessionHandle]:
        return list(self._sessions.values())

    def count(self) -> int:
        return len(self._sessions)

    async def stop(self, session_name: str) -> bool:
        """Stop one session. Returns whether its process is gone. Idempotent."""
        handle = self._sessions.get(session_name)
        if handle is None:
            return True
        await asyncio.to_thread(handle.stop)
        self.release(handle)
        return not handle.is_alive()

    async def stop_all(self) -> None:
        """Stop every session, concurrently. Used on application shutdown."""
        handles = list(self._sessions.values())
        await asyncio.gather(*(asyncio.to_thread(handle.stop) for handle in handles))
        for handle in handles:
            self.release(handle)

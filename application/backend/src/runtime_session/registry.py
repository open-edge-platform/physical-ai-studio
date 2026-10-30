"""The live runtime sessions of this API process, and which robots they hold.

In memory: a session lives and dies with its websocket in this process, so
nothing outlives a restart. The device lock (``claim_robot``) already keeps two
holders off one robot; the registry names the holder, so a form, a delete or
another session gets a ``device_in_use`` error that says which session to close.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import contextmanager
from typing import TYPE_CHECKING

from exceptions import DeviceInUseError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from runtime_session.session import RuntimeSession


class RuntimeSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, RuntimeSession] = {}
        self._robots: dict[UUID, RuntimeSession] = {}
        self._watchers: set[asyncio.Queue[None]] = set()

    def add(self, session: RuntimeSession) -> None:
        self._sessions[session.id] = session
        self.changed()

    def remove(self, session: RuntimeSession) -> None:
        self._sessions.pop(session.id, None)
        self.release_robots(session)
        self.changed()

    @contextmanager
    def watch(self) -> Iterator[asyncio.Queue[None]]:
        """A queue that gets an item whenever the session list may have changed; bursts coalesce."""
        changes: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._watchers.add(changes)
        try:
            yield changes
        finally:
            self._watchers.discard(changes)

    def changed(self) -> None:
        """Wake the watchers. Call when a session's listing changed (loaded or unloaded something)."""
        for changes in self._watchers:
            if changes.empty():
                changes.put_nowait(None)

    def get(self, session_id: str) -> RuntimeSession | None:
        return self._sessions.get(session_id)

    def sessions(self) -> list[RuntimeSession]:
        return list(self._sessions.values())

    def claim_robots(self, session: RuntimeSession, robots: dict[UUID, str]) -> None:
        """Claim every robot (id to display name) for ``session``, or none of them.

        Raises:
            DeviceInUseError: Another session holds one of them.
        """
        for robot_id, name in robots.items():
            holder = self._robots.get(robot_id)
            if holder is not None and holder is not session:
                raise DeviceInUseError(device=f"Robot {name!r}", holder=holder.label)
        for robot_id in robots:
            self._robots[robot_id] = session

    def release_robots(self, session: RuntimeSession) -> None:
        for robot_id in [robot_id for robot_id, holder in self._robots.items() if holder is session]:
            del self._robots[robot_id]

    def holder_of_robot(self, robot_id: UUID) -> RuntimeSession | None:
        return self._robots.get(robot_id)

    def require_robot_free(self, robot_id: UUID, name: str) -> None:
        """Raises ``DeviceInUseError`` when a session holds the robot."""
        if (holder := self._robots.get(robot_id)) is not None:
            raise DeviceInUseError(device=f"Robot {name!r}", holder=holder.label)

    async def until_closed(self, within: float) -> None:
        """Wait for every session that is closing to let go of its devices; give up after ``within`` seconds.

        A page that opens a session usually just closed one on the same devices, which
        takes a moment to disconnect them.
        """
        closing = [session.finished.wait() for session in self.sessions() if session.closed.is_set()]
        if closing:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.gather(*closing), within)

    async def close_all(self) -> None:
        await asyncio.gather(*(session.stop() for session in self.sessions()))

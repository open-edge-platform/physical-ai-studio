from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from uuid import uuid4

from runtime_session.registry import RuntimeSessionRegistry


def _closing_session(session_id: str) -> SimpleNamespace:
    closed, finished = asyncio.Event(), asyncio.Event()
    closed.set()
    return SimpleNamespace(id=session_id, label=session_id, closed=closed, finished=finished)


async def test_a_load_waits_only_for_the_closing_session_that_holds_its_robot() -> None:
    parking, other = uuid4(), uuid4()
    registry = RuntimeSessionRegistry()
    session = _closing_session("parking")
    registry.add(session)  # type: ignore[arg-type]
    registry.claim_robots(session, {parking: "reBot"})  # type: ignore[arg-type]

    started = time.monotonic()
    await registry.until_closed(within=5.0, holding=[other])
    assert time.monotonic() - started < 0.5

    asyncio.get_running_loop().call_later(0.3, session.finished.set)  # the park ends
    started = time.monotonic()
    await registry.until_closed(within=5.0, holding=[parking])
    assert 0.25 < time.monotonic() - started < 2.0

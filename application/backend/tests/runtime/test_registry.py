from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from exceptions import RuntimeSessionBusyError
from runtime.ids import runtime_session_name
from runtime.registry import RuntimeSessionRegistry


class _FakeHandle:
    def __init__(self, session_name: str | None = None) -> None:
        self.session_name = session_name or runtime_session_name(uuid4())
        self.follower_name = "follower"
        self.pid = 1234
        self.stopping = False
        self.alive = True
        self.stops = 0

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.stops += 1
        self.stopping = True
        self.alive = False


def test_a_live_session_makes_the_follower_busy() -> None:
    registry = RuntimeSessionRegistry()
    first = _FakeHandle()
    second = _FakeHandle(first.session_name)
    asyncio.run(registry.acquire(first))  # type: ignore[arg-type]

    with pytest.raises(RuntimeSessionBusyError):
        asyncio.run(registry.acquire(second))  # type: ignore[arg-type]

    assert registry.get(first.session_name) is first
    assert first.stops == 0


def test_a_stopping_session_is_waited_for_instead_of_busy() -> None:
    """A page refresh: the old websocket closed and its worker is still tearing down."""
    registry = RuntimeSessionRegistry()
    first = _FakeHandle()
    second = _FakeHandle(first.session_name)
    asyncio.run(registry.acquire(first))  # type: ignore[arg-type]
    first.stopping = True

    asyncio.run(registry.acquire(second))  # type: ignore[arg-type]

    assert first.stops == 1
    assert registry.get(first.session_name) is second


def test_a_dead_session_does_not_hold_the_follower() -> None:
    registry = RuntimeSessionRegistry()
    first = _FakeHandle()
    asyncio.run(registry.acquire(first))  # type: ignore[arg-type]
    first.alive = False

    second = _FakeHandle(first.session_name)
    asyncio.run(registry.acquire(second))  # type: ignore[arg-type]

    assert registry.get(first.session_name) is second


def test_release_only_drops_the_handle_that_holds_the_slot() -> None:
    registry = RuntimeSessionRegistry()
    first = _FakeHandle()
    first.stopping = True
    second = _FakeHandle(first.session_name)
    asyncio.run(registry.acquire(first))  # type: ignore[arg-type]
    asyncio.run(registry.acquire(second))  # type: ignore[arg-type]

    registry.release(first)  # type: ignore[arg-type]

    assert registry.get(first.session_name) is second


def test_stop_all_stops_every_session() -> None:
    registry = RuntimeSessionRegistry()
    handles = [_FakeHandle(), _FakeHandle()]
    for handle in handles:
        asyncio.run(registry.acquire(handle))  # type: ignore[arg-type]

    asyncio.run(registry.stop_all())

    assert [handle.stops for handle in handles] == [1, 1]
    assert registry.count() == 0

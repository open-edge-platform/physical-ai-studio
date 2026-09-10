from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

from runtime.contract import SetFollowerSourceCommand
from runtime.hosts import session_worker

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met in time")


class _FakeServer:
    """Just enough of RuntimeZenohServer for the subscriber watcher."""

    def __init__(self, *, subscribers: bool = True) -> None:
        self._subscribers = subscribers
        self._lock = threading.Lock()
        self.metadata_writes: list[dict[str, Any]] = []
        self.events: list[Any] = []
        self.polls = 0

    def has_matching_subscribers(self) -> bool:
        with self._lock:
            self.polls += 1
            return self._subscribers

    def set_subscribers(self, present: bool) -> None:
        with self._lock:
            self._subscribers = present

    def update_metadata(self, **values: Any) -> None:
        with self._lock:
            self.metadata_writes.append(values)

    def emit(self, event: Any, *, fatal: bool = False) -> None:
        self.events.append(event)


class _FakeSession:
    def __init__(self) -> None:
        self.commands: list[Any] = []
        self.finalized = 0

    def apply(self, command: Any) -> None:
        self.commands.append(command)

    def finalize_recording(self) -> None:
        self.finalized += 1


@contextmanager
def _running_watcher(
    server: _FakeServer,
    session: _FakeSession,
    idle_timeout_s: float,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[threading.Event]:
    monkeypatch.setattr(session_worker, "_IDLE_POLL_INTERVAL_S", 0.01)
    stop = threading.Event()
    thread = threading.Thread(
        target=session_worker._watch_subscribers,
        args=(server, session, idle_timeout_s, stop),
        daemon=True,
    )
    thread.start()
    try:
        yield stop
    finally:
        stop.set()
        thread.join(timeout=2.0)


def test_a_steady_session_never_rewrites_its_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watcher polls at 10Hz. Publishing per tick would make every poll a write."""
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 5.0, monkeypatch):
        _wait_until(lambda: server.polls > 20)

    assert server.metadata_writes == []


def test_losing_the_last_subscriber_publishes_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 5.0, monkeypatch):
        server.set_subscribers(False)
        _wait_until(lambda: bool(server.metadata_writes))

    write = server.metadata_writes[0]
    assert write["attached"] is False
    # Wall clock, so a reader can compare it against its own now.
    assert write["idle_deadline"] == pytest.approx(time.time() + 5.0, abs=2.0)


def test_an_abandoned_session_publishes_its_deadline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Staying away is the long-lived state, so it must not write on every poll."""
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 5.0, monkeypatch):
        server.set_subscribers(False)
        _wait_until(lambda: bool(server.metadata_writes))
        settled = server.polls
        _wait_until(lambda: server.polls > settled + 20)

    assert len(server.metadata_writes) == 1


def test_a_returning_client_clears_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 5.0, monkeypatch):
        server.set_subscribers(False)
        _wait_until(lambda: bool(server.metadata_writes))
        server.set_subscribers(True)
        _wait_until(lambda: len(server.metadata_writes) >= 2)
        settled = server.polls
        _wait_until(lambda: server.polls > settled + 20)

    assert server.metadata_writes[-1] == {"attached": True, "idle_deadline": None}
    assert len(server.metadata_writes) == 2


def test_losing_the_last_subscriber_still_holds_the_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metadata write must not have displaced what the watcher already did."""
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 5.0, monkeypatch):
        server.set_subscribers(False)
        _wait_until(lambda: session.finalized > 0)

    assert session.finalized == 1
    assert [type(command) for command in session.commands] == [SetFollowerSourceCommand]
    assert session.commands[0].follower_source == "hold"


def test_an_idle_session_still_shuts_itself_down(monkeypatch: pytest.MonkeyPatch) -> None:
    server, session = _FakeServer(), _FakeSession()

    with _running_watcher(server, session, 0.05, monkeypatch) as stop:
        server.set_subscribers(False)
        _wait_until(lambda: stop.is_set())

    assert [event.data.reason for event in server.events] == ["idle_timeout"]

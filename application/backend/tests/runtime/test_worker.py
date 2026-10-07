"""Run a real RuntimeSessionWorker process with fake devices."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from runtime.contract import AckEvent, SaveEpisodeCommand, SetFollowerSourceCommand, StateEvent
from runtime.handle import RuntimeProcessError, RuntimeSessionHandle
from runtime.ids import runtime_session_name
from tests.runtime.test_session import _document, _document_with_leader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from runtime.contract import RuntimeEvent


def _handle(document: dict[str, Any], stop_event: Any | None = None) -> RuntimeSessionHandle:
    follower_id = uuid4()
    return RuntimeSessionHandle(
        runtime_session_name(follower_id),
        follower_id=follower_id,
        document=document,
        follower_name="follower",
        leader_name=None,
        stop_event=stop_event if stop_event is not None else mp.Event(),
    )


@pytest.fixture
def started() -> Iterator[Callable[..., RuntimeSessionHandle]]:
    handles: list[RuntimeSessionHandle] = []

    def start(document: dict[str, Any], stop_event: Any | None = None) -> RuntimeSessionHandle:
        handle = _handle(document, stop_event)
        handles.append(handle)
        handle.start()
        return handle

    yield start
    for handle in handles:
        handle.stop()


def _next_event(
    handle: RuntimeSessionHandle, predicate: Callable[[RuntimeEvent], bool], timeout: float = 10.0
) -> RuntimeEvent:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = handle.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue
        if predicate(event):
            return event
    raise AssertionError("expected event did not arrive")


def test_the_worker_reports_a_connected_state(started: Callable[..., RuntimeSessionHandle]) -> None:
    handle = started(_document())

    handle.wait_until_ready()

    event = _next_event(handle, lambda event: isinstance(event, StateEvent))
    assert isinstance(event, StateEvent)
    assert event.data.connected
    assert event.data.follower_source == "hold"
    assert handle.status == "running"
    assert handle.pid is not None


def test_a_command_crosses_the_process_boundary(started: Callable[..., RuntimeSessionHandle]) -> None:
    handle = started(_document_with_leader())
    handle.wait_until_ready()

    handle.apply(SetFollowerSourceCommand(follower_source="teleop"))

    _next_event(handle, lambda event: isinstance(event, StateEvent) and event.data.follower_source == "teleop")


def test_a_request_is_answered_with_an_ack(started: Callable[..., RuntimeSessionHandle]) -> None:
    handle = started(_document())
    handle.wait_until_ready()

    handle.apply(SaveEpisodeCommand(request_id="request-7"))

    ack = _next_event(handle, lambda event: isinstance(event, AckEvent))
    assert isinstance(ack, AckEvent)
    assert ack.data.request_id == "request-7"
    # Nothing is recording, so the save fails -- and says so rather than hanging.
    assert ack.data.ok is False
    assert ack.data.error


def test_stop_tears_the_devices_down(started: Callable[..., RuntimeSessionHandle], tmp_path: Path) -> None:
    marker = tmp_path / "follower-disconnected"
    handle = started(_document(disconnect_marker=str(marker)))
    handle.wait_until_ready()

    handle.stop()

    assert not handle.is_alive()
    assert handle.status == "stopped"
    assert marker.exists()


def test_the_application_stop_event_ends_the_worker(
    started: Callable[..., RuntimeSessionHandle], tmp_path: Path
) -> None:
    marker = tmp_path / "follower-disconnected"
    app_stop = mp.Event()
    handle = started(_document(disconnect_marker=str(marker)), app_stop)
    handle.wait_until_ready()

    app_stop.set()

    deadline = time.monotonic() + 10
    while handle.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not handle.is_alive()
    assert marker.exists()


def test_a_connect_failure_is_reported(started: Callable[..., RuntimeSessionHandle]) -> None:
    handle = started(_document(connect_error="serial port is busy"))

    with pytest.raises(RuntimeProcessError, match="serial port is busy"):
        handle.wait_until_ready()
    assert handle.status == "error"


def test_a_setup_failure_is_reported(started: Callable[..., RuntimeSessionHandle]) -> None:
    document = _document()
    document["init_args"]["robot"]["class_path"] = "tests.runtime.fakes.DoesNotExist"
    handle = started(document)

    with pytest.raises(RuntimeProcessError):
        handle.wait_until_ready()


def test_a_handle_stopped_before_start_never_spawns() -> None:
    handle = _handle(_document())
    handle.stop()

    with pytest.raises(RuntimeProcessError):
        handle.start()
    assert handle.pid is None


def test_stop_is_idempotent(started: Callable[..., RuntimeSessionHandle]) -> None:
    handle = started(_document())
    handle.wait_until_ready()

    handle.stop()
    handle.stop()

    assert not handle.is_alive()

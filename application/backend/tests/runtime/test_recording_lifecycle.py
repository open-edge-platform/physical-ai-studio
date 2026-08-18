from __future__ import annotations

import asyncio
import threading
import time

import pytest

from runtime.callbacks.recording import RecordingState
from runtime.command_thread import CommandWorker
from runtime.contract import AckData, DiscardEpisodeCommand, QueueEventSink, SaveEpisodeCommand
from runtime.hosts.session_worker import _watch_subscribers
from runtime.session import RuntimeSession
from tests.runtime.test_session import _document, _document_with_cameras


class _FakeMutation:
    def __init__(self) -> None:
        self.saved = 0
        self.discarded = 0
        self.torn_down = 0
        self.has_mutation = False

    def save_episode(self) -> None:
        self.has_mutation = True
        self.saved += 1

    def discard_buffer(self) -> None:
        self.discarded += 1

    def teardown(self) -> None:
        self.torn_down += 1


class _FakeServer:
    def __init__(self) -> None:
        self.matching = True
        self.events: list = []

    def has_matching_subscribers(self) -> bool:
        return self.matching

    def emit(self, event: object, *, fatal: bool = False) -> None:
        self.events.append(event)


class _FakeIdleSession:
    def __init__(self) -> None:
        self.is_recording = False
        self.commands: list = []

    def apply(self, command: object) -> None:
        self.commands.append(command)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition was not met in time")


def test_save_and_discard_keep_their_order() -> None:
    worker = CommandWorker()
    order: list[str] = []

    def save() -> None:
        time.sleep(0.05)
        order.append("save")

    def discard() -> None:
        order.append("discard")

    try:
        worker.submit("save_episode", save, request_id="save-1")
        worker.submit("discard_episode", discard, request_id="discard-1")
        save_ack = worker.wait("save-1", timeout=2.0)
        discard_ack = worker.wait("discard-1", timeout=2.0)
    finally:
        worker.shutdown(timeout=2.0)

    assert order == ["save", "discard"]
    assert save_ack == AckData(request_id="save-1", ok=True)
    assert discard_ack == AckData(request_id="discard-1", ok=True)


def test_save_discard_save_produces_two_episodes() -> None:
    mutation = _FakeMutation()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    mutation.save_episode()
    recording.mark_saved()
    recording.start("pick")
    mutation.discard_buffer()
    recording.mark_discarded()
    recording.start("pick")
    mutation.save_episode()
    recording.mark_saved()

    assert mutation.saved == 2
    assert mutation.discarded == 1
    assert mutation.has_mutation is True
    assert recording.episodes_recorded == 2


def test_the_idle_countdown_pauses_while_recording() -> None:
    server = _FakeServer()
    session = _FakeIdleSession()
    session.is_recording = True
    server.matching = False
    stop = threading.Event()
    thread = threading.Thread(
        target=_watch_subscribers,
        args=(server, session, 0.2, stop),
        daemon=True,
    )
    thread.start()
    time.sleep(0.5)
    still_running = thread.is_alive()
    stop.set()
    thread.join(timeout=2.0)

    assert still_running
    assert session.commands


def test_explicit_stop_mid_recording_finalizes_the_dataset() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")

    asyncio.run(session.teardown())

    assert mutation.torn_down == 1
    assert session._recording.dataset_loaded is False


def test_idle_exit_after_save_finalizes_the_cache() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")
    mutation.save_episode()
    session._recording.mark_saved()

    asyncio.run(session.teardown())

    assert mutation.saved == 1
    assert mutation.torn_down == 1


def test_loading_a_dataset_does_not_stall_the_loop() -> None:
    session = RuntimeSession(_document_with_cameras("front"), event_sink=QueueEventSink())
    stop = threading.Event()
    asyncio.run(session.setup())
    session._preconnect_devices()
    thread = threading.Thread(target=session.run, args=(stop,), daemon=True)
    thread.start()
    _wait_until(lambda: session.ready.is_set())
    follower = session._follower
    assert follower is not None

    def sleeping_load() -> None:
        time.sleep(1.0)
        session._recording.attach_mutation(_FakeMutation())

    started = len(follower.sent_actions)
    session._command_worker.submit("load_dataset", sleeping_load)
    _wait_until(lambda: session._recording.dataset_loaded, timeout=3.0)
    sent_during_copy = len(follower.sent_actions) - started
    stop.set()
    thread.join(timeout=3.0)
    asyncio.run(session.teardown())

    assert sent_during_copy >= 10


def test_handle_request_acks_save_and_discard_with_their_request_ids() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")

    session.handle_request(SaveEpisodeCommand(request_id="save-1"))
    session._recording.start("pick")
    session.handle_request(DiscardEpisodeCommand(request_id="discard-1"))
    asyncio.run(session.teardown())

    assert mutation.saved == 1
    assert mutation.discarded == 1


def test_save_episode_stops_recording_before_the_mutation_writes() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    recording = session._recording

    class _OrderCheckingMutation(_FakeMutation):
        def save_episode(self) -> None:
            assert recording.is_recording is False
            super().save_episode()

    mutation = _OrderCheckingMutation()
    recording.attach_mutation(mutation)
    recording.start("pick")
    session._save_episode()

    assert mutation.saved == 1
    assert recording.is_recording is False
    assert recording.episodes_recorded == 1

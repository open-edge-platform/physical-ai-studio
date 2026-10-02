from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

from runtime.callbacks.recording import RecordingState
from runtime.command_thread import CommandWorker, _Job
from runtime.contract import AckData, DiscardEpisodeCommand, QueueEventSink, SaveEpisodeCommand, StateEvent
from runtime.session import RuntimeSession
from tests.runtime.test_session import _document, _document_with_cameras


class _FakeMutation:
    def __init__(self, save_delay: float = 0.0) -> None:
        self.saved = 0
        self.discarded = 0
        self.torn_down = 0
        self.has_mutation = False
        self.order: list[str] = []
        self._save_delay = save_delay

    def save_episode(self) -> None:
        time.sleep(self._save_delay)
        self.has_mutation = True
        self.saved += 1
        self.order.append("save")

    def discard_buffer(self) -> None:
        self.discarded += 1
        self.order.append("discard")

    def teardown(self) -> None:
        self.torn_down += 1
        self.order.append("teardown")


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


def test_explicit_stop_mid_recording_finalizes_the_dataset() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")

    asyncio.run(session.teardown())

    assert mutation.torn_down == 1
    assert session._recording.dataset_loaded is False


def test_teardown_after_save_finalizes_the_cache() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")
    mutation.save_episode()
    session._recording.mark_saved()

    asyncio.run(session.teardown())

    assert mutation.saved == 1
    assert mutation.torn_down == 1


def test_finalizing_resets_the_session_episode_count() -> None:
    """Those episodes are in the dataset now, so the UI must not add them twice."""
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")
    session._recording.mark_saved()
    assert session._recording.episodes_recorded == 1

    session._finalize_recording()

    assert session._recording.episodes_recorded == 0
    assert session._recording.dataset_loaded is False


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


class _FailingSaveMutation(_FakeMutation):
    def save_episode(self) -> None:
        raise RuntimeError("video encoding failed")


def _last_state(sink: QueueEventSink) -> StateEvent | None:
    latest: StateEvent | None = None
    while True:
        try:
            event = sink.get_nowait()
        except queue.Empty:
            return latest
        if isinstance(event, StateEvent):
            latest = event


def test_a_job_accepted_before_shutdown_is_never_dropped() -> None:
    """submit and shutdown race for the queue, and the job must not land behind the sentinel.

    Enqueueing outside the lock let shutdown insert _SHUTDOWN between the
    closed check and the put, so the worker returned without running a job it
    had already accepted. Slowing the job's own enqueue widens that window to
    something a test can observe; under the lock, shutdown simply waits.
    """
    worker = CommandWorker()
    ran = threading.Event()
    enqueueing = threading.Event()
    real_put = worker._jobs.put

    def slow_put(item, *args, **kwargs):
        if isinstance(item, _Job):
            enqueueing.set()
            time.sleep(0.2)
        return real_put(item, *args, **kwargs)

    worker._jobs.put = slow_put

    def stop_once_submitting() -> None:
        enqueueing.wait(2.0)
        worker.shutdown(timeout=2.0)

    stopper = threading.Thread(target=stop_once_submitting)
    stopper.start()
    worker.submit("save_episode", ran.set)
    stopper.join(timeout=5.0)

    assert ran.wait(2.0), "a job accepted before shutdown was dropped behind the sentinel"


def test_teardown_leaves_the_mutation_to_a_worker_that_did_not_drain(monkeypatch) -> None:
    """Finalizing on top of a running save stops the image writer mid-encode."""
    monkeypatch.setattr("runtime.session.RECORDING_TEARDOWN_TIMEOUT_S", 0.05)
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FakeMutation(save_delay=0.5)
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")
    session._command_worker.submit("save_episode", session._save_episode)

    asyncio.run(session.teardown())

    assert mutation.torn_down == 0
    _wait_until(lambda: mutation.saved == 1)
    assert mutation.order == ["save"]


def test_a_failed_save_publishes_the_stopped_state() -> None:
    """The ack carries the error, but the browser only moves on a state event."""
    sink = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=sink)
    asyncio.run(session.setup())
    mutation = _FailingSaveMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")

    with pytest.raises(RuntimeError, match="video encoding failed"):
        session._save_episode()

    state = _last_state(sink)
    assert state is not None
    assert state.data.is_recording is False
    assert session._recording.episodes_recorded == 0


def test_discard_clears_the_buffer_after_a_failed_save() -> None:
    """Discard is the recovery path, so it cannot require an open episode."""
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    mutation = _FailingSaveMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")
    with pytest.raises(RuntimeError):
        session._save_episode()

    session._discard_episode()

    assert mutation.discarded == 1
    assert session._recording.is_recording is False


def test_teardown_discards_an_open_episode_before_finalizing() -> None:
    """An open buffer is never copied back, so drop it deliberately and say so."""
    sink = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=sink)
    asyncio.run(session.setup())
    mutation = _FakeMutation()
    session._recording.attach_mutation(mutation)
    session._recording.start("pick")

    asyncio.run(session.teardown())

    assert mutation.order == ["discard", "teardown"]
    state = _last_state(sink)
    assert state is not None
    assert state.data.is_recording is False

import shutil
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from internal_datasets.mutations.recording_mutation import RecordingMutation
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import SaveEpisodeCommand, StartEpisodeCommand, StatusMessage
from workers.dataset_worker import Recorder, lowest_free_bytes


@pytest.fixture
def recorder():
    store = Store.create(FeatureSpec((), {"cam": ImageSpec(2, 2)}), uuid4().hex)
    for key in (STATE, ACTION):
        store.write(key, [], timestamp=1.0)
    store.write_observation("cam", np.zeros((2, 2, 3), dtype=np.uint8), timestamp=1.0)
    recorder = Recorder(MagicMock(), store, publish=MagicMock(), episodes_recorded=1)
    recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
    yield recorder
    store.unlink()


def test_missing_camera_discards_only_the_open_episode(recorder):
    recorder.record_frame()
    recorder.store.invalidate_observation("cam")
    recorder.record_frame()
    recorder.record_frame()

    recorder.mutation.discard_buffer.assert_called_once()
    assert recorder.status().episodes_recorded == 1
    assert not recorder.status().is_recording
    status = recorder.publish.call_args.args[0]
    assert 'camera "cam"' in status.error
    assert "Episode discarded" in status.error
    with pytest.raises(RuntimeSessionCommandError, match="No episode"):
        recorder.save(SaveEpisodeCommand(request_id="save"))
    recorder.mutation.save_episode.assert_not_called()


@pytest.mark.parametrize("state", ["starting", "failed"])
def test_camera_loss_between_recording_ticks_discards_the_episode(recorder, state):
    recorder.record_frame()
    # The slot can already have a fresh frame again when the queued status arrives.
    recorder.camera_status(StatusMessage(component="environment/cam", state=state, error="camera lost"))
    recorder.record_frame()

    recorder.mutation.discard_buffer.assert_called_once()
    assert not recorder.status().is_recording
    assert recorder.publish.call_args.args[0].error is not None

    recorder.start(StartEpisodeCommand(request_id="restart", task="pick"))
    assert recorder.publish.call_args.args[0].error is None
    recorder.record_frame()
    recorder.record_frame()
    assert recorder.save(SaveEpisodeCommand(request_id="save")).episodes_recorded == 2


@pytest.mark.parametrize(
    ("component", "state"),
    [("environment/cam", "running"), ("environment/follower", "starting"), ("environment/other", "failed")],
)
def test_unrelated_status_does_not_discard_an_episode(recorder, component, state):
    recorder.camera_status(StatusMessage(component=component, state=state))

    recorder.mutation.discard_buffer.assert_not_called()
    assert recorder.status().is_recording


def test_camera_loss_while_idle_does_not_discard_or_notify(recorder):
    recorder.discard()
    recorder.publish.reset_mock()
    recorder.mutation.discard_buffer.reset_mock()

    recorder.camera_status(StatusMessage(component="environment/cam", state="starting"))

    recorder.publish.assert_not_called()
    recorder.mutation.discard_buffer.assert_not_called()


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _frame(recorder, timestamp: float) -> None:
    recorder.store.write_observation("cam", np.zeros((2, 2, 3), dtype=np.uint8), timestamp=timestamp)


def _messages(recorder) -> list[str]:
    return [call.args[0].error for call in recorder.publish.call_args_list if call.args[0].error]


@pytest.mark.parametrize("fps", [10, 30])
def test_a_stalled_camera_warns_once_per_stall_after_0_2_s(recorder, fps):
    clock = recorder.clock = Clock()
    period = 1 / fps

    def tick(new_frame: bool) -> None:
        clock.now += period
        if new_frame:
            _frame(recorder, clock.now)
        recorder.record_frame()

    tick(True)
    tick(False)  # one frame on two ticks: normal when the rates are close
    tick(True)
    for _ in range(round(0.2 * fps) - 1):
        tick(False)
    assert _messages(recorder) == []  # not yet 0.2 s stale
    tick(False)
    assert _messages(recorder) == [
        'Camera "cam" sent no new frame for 0.2 s, so this episode repeats one frame. '
        "Check the camera, and discard the episode if the scene moved."
    ]
    for _ in range(fps):  # the same stall, a second longer: no second warning
        tick(False)
    assert len(_messages(recorder)) == 1
    assert recorder.status().is_recording  # a warning, not a discard

    tick(True)  # the stall ends; the next one warns again, once
    for _ in range(round(0.2 * fps) + 3):
        tick(False)
    assert len(_messages(recorder)) == 2


def test_an_episode_does_not_start_on_a_nearly_full_disk(recorder):
    recorder.discard()
    recorder.free_bytes = lambda: 1024**3
    with pytest.raises(RuntimeSessionCommandError, match=r"Only 1\.0 GB is free") as refused:
        recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
    assert refused.value.error_code == "low_disk"
    assert not recorder.status().is_recording


def test_a_failing_disk_probe_does_not_block_a_start(recorder):
    def broken() -> int:
        raise OSError("statvfs failed")

    recorder.discard()
    recorder.free_bytes = broken
    assert recorder.start(StartEpisodeCommand(request_id="start", task="pick")).is_recording


def test_a_failed_frame_tells_the_user_the_episode_was_discarded(recorder):
    recorder.mutation.add_frame.side_effect = ValueError("a frame of the wrong shape")
    recorder.record_frame()

    recorder.mutation.discard_buffer.assert_called_once()
    assert not recorder.status().is_recording
    assert _messages(recorder) == [
        "Episode discarded because recording a frame failed: a frame of the wrong shape. Start a new episode."
    ]


@pytest.mark.parametrize(
    ("fails", "says"),
    [
        ("finalize", "Finalizing failed (No space left on device). What was recorded is left in /cache/3f2a"),
        ("overwrite", "Copying back failed (No space left on device). The recorded episodes are left in /cache/3f2a"),
    ],
)
def test_a_failed_teardown_names_the_cache_it_left(fails, says):
    source, cache = MagicMock(), MagicMock()
    cache.path = "/cache/3f2a"
    {"finalize": cache.finalize, "overwrite": source.overwrite}[fails].side_effect = OSError("No space left on device")
    mutation = RecordingMutation(source, cache)
    mutation.save_episode()
    with pytest.raises(RuntimeError) as failed:
        mutation.teardown()
    assert str(failed.value) == says
    cache.delete.assert_not_called()  # the cache stays, for recovery


def test_a_lost_episode_whose_frames_could_not_be_cleared_blocks_the_next_start(recorder):
    recorder.mutation.add_frame.side_effect = ValueError("a frame of the wrong shape")
    recorder.mutation.discard_buffer.side_effect = OSError("the episode's files are gone")
    recorder.record_frame()

    assert not recorder.status().is_recording
    message = _messages(recorder)[-1]
    assert message.startswith("Episode discarded because recording a frame failed: a frame of the wrong shape.")
    assert "Clearing its frames failed too (the episode's files are gone)" in message
    assert "Start a new episode" not in message
    with pytest.raises(RuntimeSessionCommandError, match="could not be cleared") as refused:
        recorder.start(StartEpisodeCommand(request_id="again", task="pick"))
    assert refused.value.error_code == "buffer_not_cleared"

    recorder.mutation.discard_buffer.side_effect = None  # clearing works again: the start clears first
    assert recorder.start(StartEpisodeCommand(request_id="again", task="pick")).is_recording


def test_a_slow_write_is_not_taken_for_a_stalled_camera(recorder):
    clock = recorder.clock = Clock()
    slow = [False]

    def write(*_args) -> None:
        if slow[0]:
            clock.now += 0.5  # writing this frame takes long; the camera is fine

    recorder.mutation.add_frame.side_effect = write
    for _ in range(6):
        clock.now += 1 / 30
        _frame(recorder, clock.now)  # a new frame, written quickly
        slow[0] = False
        recorder.record_frame()
        clock.now += 1 / 30  # the same frame read once more a tick later, an ordinary duplicate, written slowly
        slow[0] = True
        recorder.record_frame()
    assert _messages(recorder) == []


def test_one_unreadable_disk_does_not_hide_a_full_one(monkeypatch, tmp_path):
    full, broken = tmp_path / "cache", tmp_path / "dataset"
    full.mkdir()
    broken.mkdir()

    def disk_usage(path):
        if path == broken:
            raise PermissionError("no access")
        return shutil._ntuple_diskusage(10, 10 - 512, 512)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    assert lowest_free_bytes([full, broken]) == 512
    with pytest.raises(PermissionError):
        lowest_free_bytes([broken])

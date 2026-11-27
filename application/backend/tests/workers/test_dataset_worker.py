import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.recording_mutation import (
    RecordingCleanupError,
    RecordingMutation,
    RecordingNotSavedError,
    RecordingOutcomeUnknownError,
    UnconfirmedCleanupError,
    UnusedCacheCleanupError,
)
from runtime_session.bus import DEADLINE
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import DiscardEpisodeCommand, SaveEpisodeCommand, StartEpisodeCommand, StatusMessage
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


def _messages(recorder) -> list[str]:
    return [call.args[0].error for call in recorder.publish.call_args_list if call.args[0].error]


def test_an_episode_does_not_start_on_a_nearly_full_disk(recorder):
    recorder.discard()
    recorder.free_bytes = lambda: 1024**3
    with pytest.raises(RuntimeSessionCommandError, match=r"Only 1\.0 GB is free") as refused:
        recorder.start(StartEpisodeCommand(request_id="start", task="pick"))
    assert refused.value.error_code == "low_disk"
    assert not recorder.status().is_recording


def test_an_episode_does_not_open_once_the_client_stopped_waiting(recorder):
    recorder.discard()
    start = StartEpisodeCommand(request_id="start", task="pick")
    token = DEADLINE.set(time.monotonic() - 0.01)  # as the bus sets it for the request being answered
    try:
        with pytest.raises(RuntimeSessionCommandError) as refused:
            recorder.start(start)
    finally:
        DEADLINE.reset(token)
    assert refused.value.error_code == "command_timeout"
    assert not recorder.status().is_recording

    token = DEADLINE.set(time.monotonic() + 10)
    try:
        assert recorder.start(start).is_recording
    finally:
        DEADLINE.reset(token)


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
    ("fails", "kind", "says"),
    [
        (
            "finalize",
            RecordingNotSavedError,
            "Finalizing failed (No space left on device). What was recorded is left in /cache/3f2a",
        ),
        (
            "overwrite",
            RecordingOutcomeUnknownError,
            "Copying back stopped part way (No space left on device), so the dataset may be incomplete. "
            "The recorded episodes are kept in /cache/3f2a",
        ),
    ],
)
def test_a_failed_teardown_names_the_cache_it_left(fails, kind, says):
    source, cache = MagicMock(), MagicMock()
    cache.path = "/cache/3f2a"
    {"finalize": cache.finalize, "overwrite": source.overwrite}[fails].side_effect = OSError("No space left on device")
    mutation = RecordingMutation(source, cache)
    mutation.save_episode()
    with pytest.raises(kind) as failed:
        mutation.teardown()
    assert str(failed.value) == says
    cache.delete.assert_not_called()  # the cache stays, for recovery


def _lerobot_mutation(tmp_path: Path) -> tuple[RecordingMutation, Path, Path]:
    """The real mutation and copy back over two folders; lerobot's own writing is left out."""
    source, cache = tmp_path / "dataset", tmp_path / "cache"
    (cache / "meta").mkdir(parents=True)
    (cache / "meta" / "episodes.txt").write_text("episode 0")
    (cache / "data.bin").write_bytes(b"frames")
    cache_dataset = InternalLeRobotDataset(cache)
    cache_dataset.prepare_for_writing = lambda: None  # type: ignore[method-assign]
    cache_dataset.finalize = lambda: None  # type: ignore[method-assign]
    cache_dataset.save_episode = lambda: None  # type: ignore[method-assign]
    mutation = RecordingMutation(InternalLeRobotDataset(source), cache_dataset)
    return mutation, source, cache


def test_a_dataset_that_will_not_open_after_a_complete_copy_is_unknown_not_lost(tmp_path):
    mutation, source, cache = _lerobot_mutation(tmp_path)
    mutation.source_dataset.load_dataset = MagicMock(side_effect=OSError("meta/info.json is unreadable"))
    mutation.save_episode()
    with pytest.raises(RecordingOutcomeUnknownError) as failed:
        mutation.teardown()
    assert str(failed.value) == (
        f"Every file was copied to {source}, but it could not be opened (meta/info.json is unreadable). "
        f"The recorded episodes are also kept in {cache}"
    )
    assert (source / "meta" / "episodes.txt").read_text() == "episode 0"  # it landed
    assert (cache / "data.bin").exists()  # and the cache stays


def test_a_copy_back_that_stops_part_way_is_unknown_and_keeps_the_cache(tmp_path, monkeypatch):
    mutation, source, cache = _lerobot_mutation(tmp_path)

    def copies_one_file(src, dst):
        Path(dst).mkdir()
        shutil.copy(Path(src) / "data.bin", dst)
        raise OSError("No space left on device")

    monkeypatch.setattr("internal_datasets.lerobot.lerobot_dataset.shutil.copytree", copies_one_file)
    mutation.save_episode()
    with pytest.raises(RecordingOutcomeUnknownError, match=r"stopped part way .* may be incomplete"):
        mutation.teardown()
    assert (source / "data.bin").exists() and not (source / "meta").exists()
    assert (cache / "meta" / "episodes.txt").exists()


def test_a_cache_that_stays_after_nothing_was_copied_does_not_claim_a_save(tmp_path, monkeypatch):
    """A new dataset with no accepted episode: nothing is copied, so a failed cache removal is not a save."""
    mutation, source, cache = _lerobot_mutation(tmp_path)
    mutation.cache_dataset.delete = MagicMock(side_effect=PermissionError("Permission denied"))  # type: ignore[method-assign]
    with pytest.raises(UnusedCacheCleanupError) as failed:
        mutation.teardown()
    assert str(failed.value) == f"the cache {cache} could not be removed (Permission denied)"
    assert not source.exists()


def test_a_failed_accept_then_a_cache_that_stays_does_not_claim_a_save(tmp_path):
    """The real mutation: an Accept that failed is attempted, not confirmed; the cache is still copied, as on main."""
    mutation, source, cache = _lerobot_mutation(tmp_path)
    mutation.cache_dataset.save_episode = MagicMock(side_effect=OSError("No space left on device"))  # type: ignore[method-assign]
    mutation.cache_dataset.delete = MagicMock(side_effect=PermissionError("Permission denied"))  # type: ignore[method-assign]
    with pytest.raises(OSError, match="No space left"):
        mutation.save_episode()
    with pytest.raises(UnconfirmedCleanupError) as failed:
        mutation.teardown()
    assert str(failed.value) == (
        f"an Accept failed while recording, so what it holds is not confirmed, and the cache {cache} could not be "
        "removed (Permission denied)"
    )
    assert (source / "meta" / "episodes.txt").exists()  # copied back, as on main


def test_a_cache_that_stays_after_the_copy_is_a_saved_recording(tmp_path):
    mutation, source, _cache = _lerobot_mutation(tmp_path)
    mutation.cache_dataset.delete = MagicMock(side_effect=PermissionError("Permission denied"))  # type: ignore[method-assign]
    mutation.save_episode()
    with pytest.raises(RecordingCleanupError):
        mutation.teardown()
    assert (source / "meta" / "episodes.txt").exists()


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


def test_a_command_for_another_dataset_is_refused(recorder):
    recorder.dataset_id = uuid4()
    other = uuid4()

    # A relayed command that waited behind a dataset swap must not act on the new dataset.
    for command in (
        SaveEpisodeCommand(request_id="save", dataset_id=other),
        DiscardEpisodeCommand(request_id="discard", dataset_id=other),
        StartEpisodeCommand(request_id="start", task="pick", dataset_id=other),
    ):
        handle = {"save_episode": recorder.save, "discard_episode": recorder.discard, "start_episode": recorder.start}
        with pytest.raises(RuntimeSessionCommandError) as refused:
            handle[command.command](command)
        assert refused.value.error_code == "dataset_changed"
    assert recorder.status().is_recording
    recorder.mutation.discard_buffer.assert_not_called()

    recorder.discard(DiscardEpisodeCommand(request_id="discard", dataset_id=recorder.dataset_id))
    assert not recorder.status().is_recording


def test_a_command_for_an_episode_that_ended_does_not_end_the_next_one(recorder):
    recorder.discard(DiscardEpisodeCommand(request_id="end"))
    recorder.start(StartEpisodeCommand(request_id="start", task="pick", episode_id="e1"))
    recorder.discard(DiscardEpisodeCommand(request_id="user"))  # the user discards it
    recorder.start(StartEpisodeCommand(request_id="again", task="pick", episode_id="e2"))
    recorder.mutation.reset_mock()

    # The simulation's late save and discard of e1 must leave e2 alone.
    for stale in (
        SaveEpisodeCommand(request_id="save", episode_id="e1"),
        DiscardEpisodeCommand(request_id="discard", episode_id="e1"),
    ):
        with pytest.raises(RuntimeSessionCommandError) as refused:
            (recorder.save if stale.command == "save_episode" else recorder.discard)(stale)
        assert refused.value.error_code == "episode_changed"
    assert recorder.status().is_recording
    recorder.mutation.discard_buffer.assert_not_called()
    recorder.mutation.save_episode.assert_not_called()

    recorder.discard(DiscardEpisodeCommand(request_id="discard", episode_id="e2"))
    assert not recorder.status().is_recording


def test_the_status_names_the_open_episode_and_counts_its_frames(recorder):
    recorder.discard(DiscardEpisodeCommand(request_id="end-manual"))
    recorder.start(StartEpisodeCommand(request_id="s", task="t", episode_id="e1"))
    recorder.record_frame()
    recorder.record_frame()

    assert (recorder.status().episode_id, recorder.status().frames) == ("e1", 2)
    recorder.save(SaveEpisodeCommand(request_id="r", episode_id="e1"))
    assert (recorder.status().episode_id, recorder.status().frames) == (None, 0)

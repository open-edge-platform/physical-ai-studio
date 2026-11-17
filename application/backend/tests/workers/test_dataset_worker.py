from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from exceptions import RuntimeSessionCommandError
from runtime_session.store import ACTION, STATE, FeatureSpec, ImageSpec, Store
from schemas.runtime_session import SaveEpisodeCommand, StartEpisodeCommand, StatusMessage
from workers.dataset_worker import Recorder


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

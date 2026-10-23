from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from physicalai.capture import Frame
from physicalai.runtime import LifecycleEvent, TickEvent

from runtime.callbacks.recording import RecordingCallback, RecordingState
from runtime.contract import QueueEventSink
from runtime.session import RuntimeSession
from tests.runtime.fakes import FakeObservation
from tests.runtime.test_session import _document


def _tick(
    *, positions: list[float], action: list[float] | None, frames: dict | None = None, step: int = 0
) -> TickEvent:
    return TickEvent(
        session_id="test",
        step=step,
        timestamp=1.0,
        robot_state=FakeObservation(np.array(positions, dtype=np.float32), 1.0),
        camera_frames=frames or {},
        action_sent=None if action is None else np.array(action, dtype=np.float32),
        loop_duration_s=0.0,
        sleep_time_s=0.0,
        stale_obs=False,
    )


def _callback(recording: RecordingState, *, source: str = "teleop") -> RecordingCallback:
    callback = RecordingCallback(recording=recording, follower_source=lambda: source)
    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )
    return callback


def test_frames_are_written_only_while_recording() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    callback = _callback(recording)

    callback.on_tick(_tick(positions=[1.0], action=[1.0]))
    mutation.add_frame.assert_not_called()

    recording.start("pick")
    callback.on_tick(_tick(positions=[1.0], action=[1.0], step=1))
    mutation.add_frame.assert_called_once()

    mutation.reset_mock()
    mutation.save_episode()
    recording.mark_saved()
    callback.on_tick(_tick(positions=[1.0], action=[1.0], step=2))
    mutation.add_frame.assert_not_called()


def test_hold_ticks_are_not_recorded() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording, source="hold")

    callback.on_tick(_tick(positions=[1.0], action=[9.0]))

    mutation.add_frame.assert_not_called()


def test_the_recorded_action_is_the_action_sent() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording)

    callback.on_tick(_tick(positions=[1.0], action=[7.0]))

    observation, action, task = mutation.add_frame.call_args.args
    assert action == {"joint.pos": 7.0}
    assert observation["joint.pos"] == 1.0
    assert task == "pick"


def test_recorded_frames_are_rgb() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording)
    frame = Frame(
        data=np.array([[[1, 2, 3]]], dtype=np.uint8),
        timestamp=1.0,
        sequence=1,
    )

    callback.on_tick(_tick(positions=[0.0], action=[0.0], frames={"front": frame}))

    observation = mutation.add_frame.call_args.args[0]
    np.testing.assert_array_equal(observation["front"], [[[1, 2, 3]]])


async def test_the_callback_closes_the_dataset() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    await session.setup()
    assert session._recording_callback is not None
    session._recording_callback.close = MagicMock(wraps=session._recording_callback.close)

    await session.teardown()

    session._recording_callback.close.assert_called_once()


def _recorded(mutation: MagicMock) -> list[tuple[float, float]]:
    """Return (observed position, action) for each written frame."""
    return [(call.args[0]["joint.pos"], call.args[1]["joint.pos"]) for call in mutation.add_frame.call_args_list]


def test_observed_frames_take_the_next_measured_state_as_their_action() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording, source="observe")

    for step, position in enumerate([1.0, 2.0, 3.0]):
        callback.on_tick(_tick(positions=[position], action=None, step=step))

    # The third tick is still buffered: it has no next state yet.
    assert _recorded(mutation) == [(1.0, 2.0), (2.0, 3.0)]
    assert all(call.args[2] == "pick" for call in mutation.add_frame.call_args_list)


def test_observed_frames_keep_their_own_cameras() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording, source="observe")

    def frame(value: int) -> Frame:
        return Frame(data=np.full((1, 1, 3), value, dtype=np.uint8), timestamp=1.0, sequence=value)

    callback.on_tick(_tick(positions=[1.0], action=None, frames={"front": frame(1)}))
    callback.on_tick(_tick(positions=[2.0], action=None, frames={"front": frame(2)}, step=1))

    np.testing.assert_array_equal(mutation.add_frame.call_args.args[0]["front"], [[[1, 1, 1]]])


def test_save_drops_the_trailing_observed_frame() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording, source="observe")
    callback.on_tick(_tick(positions=[1.0], action=None))
    callback.on_tick(_tick(positions=[2.0], action=None, step=1))

    recording.stop_episode()
    recording.mark_saved()
    recording.start("place")
    callback.on_tick(_tick(positions=[10.0], action=None, step=2))
    callback.on_tick(_tick(positions=[11.0], action=None, step=3))

    # The frame at 2.0 never had a next state inside its episode.
    assert _recorded(mutation) == [(1.0, 2.0), (10.0, 11.0)]


def test_discard_clears_the_buffered_observed_frame() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording, source="observe")
    callback.on_tick(_tick(positions=[1.0], action=None))

    assert recording.current_mutation() is mutation
    recording.mark_discarded()
    recording.start("pick")
    callback.on_tick(_tick(positions=[5.0], action=None, step=1))
    callback.on_tick(_tick(positions=[6.0], action=None, step=2))

    assert _recorded(mutation) == [(5.0, 6.0)]


def test_hold_ticks_drop_the_buffered_observed_frame() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    source = {"value": "observe"}
    callback = RecordingCallback(recording=recording, follower_source=lambda: source["value"])
    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )

    callback.on_tick(_tick(positions=[1.0], action=None))
    source["value"] = "hold"
    callback.on_tick(_tick(positions=[1.5], action=[1.5], step=1))
    source["value"] = "observe"
    callback.on_tick(_tick(positions=[4.0], action=None, step=2))
    callback.on_tick(_tick(positions=[5.0], action=None, step=3))

    assert _recorded(mutation) == [(4.0, 5.0)]


def test_an_episode_never_mixes_sent_and_observed_actions() -> None:
    mutation = MagicMock()
    recording = RecordingState()
    recording.attach_mutation(mutation)
    recording.start("pick")
    callback = _callback(recording)

    callback.on_tick(_tick(positions=[1.0], action=[7.0]))
    assert recording.episode_action_kind == "sent"
    callback.on_tick(_tick(positions=[2.0], action=None, step=1))
    callback.on_tick(_tick(positions=[3.0], action=None, step=2))

    assert _recorded(mutation) == [(1.0, 7.0)]

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


def _tick(*, positions: list[float], action: list[float], frames: dict | None = None, step: int = 0) -> TickEvent:
    return TickEvent(
        session_id="test",
        step=step,
        timestamp=1.0,
        robot_state=FakeObservation(np.array(positions, dtype=np.float32), 1.0),
        camera_frames=frames or {},
        action_sent=np.array(action, dtype=np.float32),
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

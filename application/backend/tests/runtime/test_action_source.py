from unittest.mock import MagicMock

import numpy as np
import pytest

from runtime.action_source import StudioActionSource
from runtime.callbacks.recording import RecordingState
from runtime.contract import (
    ErrorEvent,
    InMemoryCommandMailbox,
    QueueEventSink,
    SetFollowerSourceCommand,
    StartTaskCommand,
)

from .fakes import FakeObservation, FakeRobot


def _observation(values: list[float], timestamp: float, *, efforts: list[float] | None = None) -> FakeObservation:
    sensor_data = None if efforts is None else {"efforts": np.array(efforts, dtype=np.float32)}
    return FakeObservation(np.array(values, dtype=np.float32), timestamp, sensor_data)


def _source(*, follower: FakeRobot, leader: FakeRobot | None, fps: float = 30):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
    )
    source.connect(bus=object(), session_id="test")
    return source, mailbox, events


def test_hold_latches_the_target_on_entry() -> None:
    follower = FakeRobot([_observation([1, 2], 1), _observation([3, 4], 2)])
    source, _, _ = _source(follower=follower, leader=None)

    first = source.update(follower.get_observation(), {}, 0)
    second = source.update(follower.get_observation(), {}, 1)

    np.testing.assert_array_equal(first, [1, 2])
    np.testing.assert_array_equal(second, [1, 2])


def test_teleop_forwards_leader_positions_on_next_tick() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = FakeRobot([_observation([4, 5], 1)])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [4, 5])


def test_connect_rejects_mismatched_joint_names() -> None:
    follower = FakeRobot([_observation([0], 1)], joint_names=["follower_joint"])
    leader = FakeRobot([_observation([0], 1)], joint_names=["leader_joint"])

    with pytest.raises(ValueError, match="joint names must match"):
        _source(follower=follower, leader=leader)


def test_leader_read_error_uses_hold_and_session_survives() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [1, 2])
    assert source.follower_source == "teleop"


def test_leader_failure_is_bounded_and_emits_one_error() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, events = _source(follower=follower, leader=leader, fps=1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    state = follower.get_observation()

    for step in range(6):
        source.update(state, {}, step)

    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except Exception:
            break
    assert source.follower_source == "hold"
    assert sum(event.event == "error" for event in emitted) == 1

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(state, {}, 7)
    assert source.follower_source == "hold"


def _drain(events: QueueEventSink) -> list:
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except Exception:
            return emitted


def test_observe_sends_nothing_without_a_leader() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    source, mailbox, events = _source(follower=follower, leader=None)
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))

    action = source.update(follower.get_observation(), {}, 0)

    assert action is None
    assert source.follower_source == "observe"
    states = [event for event in _drain(events) if event.event == "state"]
    assert states[-1].data.follower_source == "observe"


def test_observe_sends_nothing_with_a_leader() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = FakeRobot([_observation([4, 5], 1)])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))

    assert source.update(follower.get_observation(), {}, 0) is None


def test_hold_after_observe_holds_the_measured_pose() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    source, mailbox, _ = _source(follower=follower, leader=None)
    source.update(_observation([1, 2], 1), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    assert source.update(_observation([1, 2], 2), {}, 1) is None
    # The external controller moves the arm while Studio observes.
    assert source.update(_observation([7, 8], 3), {}, 2) is None

    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    action = source.update(_observation([7, 8], 4), {}, 3)

    np.testing.assert_array_equal(action, [7, 8])


def test_teleop_fallback_after_observe_uses_the_measured_pose() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    source.update(_observation([1, 2], 1), {}, 0)
    source.update(_observation([7, 8], 2), {}, 1)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    action = source.update(_observation([7, 8], 3), {}, 2)

    np.testing.assert_array_equal(action, [7, 8])


def test_leader_failure_while_observing_does_not_take_the_arm() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, events = _source(follower=follower, leader=leader, fps=1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    state = follower.get_observation()

    actions = [source.update(state, {}, step) for step in range(6)]

    assert actions == [None] * 6
    assert source.follower_source == "observe"
    assert sum(event.event == "error" for event in _drain(events)) == 1


def _recording_source(kind_frame: str) -> tuple[StudioActionSource, InMemoryCommandMailbox, QueueEventSink]:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], t) for t in range(1, 20)])
    source, mailbox, events = _source(follower=follower, leader=leader)
    recording = RecordingState()
    recording.attach_mutation(MagicMock())
    recording.start("pick")
    if kind_frame == "sent":
        recording.add_frame({"joint.pos": 1.0}, {"joint.pos": 1.0})
    else:
        recording.add_observed_frame({"joint.pos": 1.0}, {"joint.pos": 1.0})
    source.bind_recording(recording)
    return source, mailbox, events


def test_switching_into_observe_is_refused_during_a_teleop_episode() -> None:
    source, mailbox, events = _recording_source("sent")
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(_observation([1, 2], 1), {}, 0)
    _drain(events)

    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    action = source.update(_observation([1, 2], 2), {}, 1)

    assert source.follower_source == "teleop"
    assert action is not None
    errors = [event for event in _drain(events) if isinstance(event, ErrorEvent)]
    assert [error.error_code for error in errors] == ["recording_source_conflict"]


def test_leaving_observe_for_studio_control_is_refused_during_an_observed_episode() -> None:
    source, mailbox, events = _recording_source("observed")
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    source.update(_observation([1, 2], 1), {}, 0)

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    mailbox.apply(StartTaskCommand(task="pick"))
    source.update(_observation([1, 2], 2), {}, 1)

    assert source.follower_source == "observe"
    errors = [event for event in _drain(events) if isinstance(event, ErrorEvent)]
    assert [error.error_code for error in errors] == ["recording_source_conflict"] * 2


def test_hold_is_always_allowed_during_an_observed_episode() -> None:
    source, mailbox, _ = _recording_source("observed")
    mailbox.apply(SetFollowerSourceCommand(follower_source="observe"))
    source.update(_observation([1, 2], 1), {}, 0)

    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    action = source.update(_observation([3, 4], 2), {}, 1)

    assert source.follower_source == "hold"
    np.testing.assert_array_equal(action, [3, 4])

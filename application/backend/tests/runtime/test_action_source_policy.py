from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from physicalai.capture import Frame
from physicalai.inference.constants import IMAGES, STATE
from physicalai.runtime import AsyncExecution, ChunkedActionQueue, LerpSmoother, PolicySource, WorkerDiedError

from runtime.action_source import StudioActionSource
from runtime.contract import (
    ErrorEvent,
    InMemoryCommandMailbox,
    QueueEventSink,
    SetFollowerSourceCommand,
    StartTaskCommand,
    StopTaskCommand,
)

from .fakes import FakeInferenceModel, FakeObservation, FakeRobot


def _observation(values: list[float], timestamp: float = 1.0) -> FakeObservation:
    return FakeObservation(np.array(values, dtype=np.float32), timestamp)


def _source(*, follower: FakeRobot | None = None, leader: FakeRobot | None = None, fps: float = 30, **kwargs):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower or FakeRobot([_observation([0.0, 0.0])]),
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
        **kwargs,
    )
    source.connect(bus=MagicMock(), session_id="test")
    return source, mailbox, events


def _drain(events: QueueEventSink) -> list:
    items = []
    while True:
        try:
            items.append(events.get_nowait())
        except Exception:
            return items


def test_re_arming_resets_before_switching_mode() -> None:
    order: list[str] = []
    source, mailbox, _ = _source()

    class _Policy:
        def set_task(self, task: str | None) -> None:
            order.append(f"set_task:{task}")

        def reset(self, *, reset_model: bool = True) -> None:
            order.append("reset")
            assert source.follower_source != "policy"

        def update(self, robot_state, camera_frames, step):
            return np.array(robot_state.joint_positions, dtype=np.float32)

        def disconnect(self) -> None:
            return None

        def to_model_input(self, robot_state, camera_frames):
            return {STATE: np.array([robot_state.joint_positions], dtype=np.float32)}

    policy = _Policy()
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    source.update(_observation([1.0, 2.0]), {}, 0)

    assert order == ["set_task:pick", "reset"]
    assert source.follower_source == "policy"


def test_re_arming_sends_an_action_from_the_current_observation() -> None:
    def predict(observation: dict) -> np.ndarray:
        state = np.asarray(observation[STATE], dtype=np.float32)
        return np.repeat(state, 4, axis=0)

    model = FakeInferenceModel(predict=predict, chunk=np.zeros((4, 2), dtype=np.float32))

    policy = PolicySource(
        model=model,
        execution=AsyncExecution(request_threshold=0.5),
        action_queue=ChunkedActionQueue(smoother=LerpSmoother()),
        task=None,
    )
    source, mailbox, _ = _source(follower=FakeRobot([_observation([1.0, 2.0])]))
    policy.connect(bus=MagicMock(), session_id="test")
    source._set_policy(policy, generation=1)

    mailbox.apply(StartTaskCommand(task="go"))
    first = source.update(_observation([1.0, 2.0]), {}, 0)
    np.testing.assert_array_almost_equal(first, [1.0, 2.0])

    mailbox.apply(StopTaskCommand())
    source.update(_observation([9.0, 8.0]), {}, 1)
    assert source.follower_source == "hold"

    mailbox.apply(StartTaskCommand(task="go"))
    rearmed = source.update(_observation([9.0, 8.0]), {}, 2)
    np.testing.assert_array_almost_equal(rearmed, [9.0, 8.0])
    policy.disconnect()


def test_a_straggler_error_on_re_arm_keeps_the_session_alive() -> None:
    source, mailbox, events = _source()

    class _Policy:
        def set_task(self, task: str | None) -> None:
            return None

        def reset(self, *, reset_model: bool = True) -> None:
            raise RuntimeError(
                "Previous inference worker is still inside the model after 12.0s. Refusing to start a second worker."
            )

        def update(self, robot_state, camera_frames, step):
            return np.array([99.0, 99.0], dtype=np.float32)

        def disconnect(self) -> None:
            return None

    source._set_policy(_Policy(), generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    action = source.update(_observation([3.0, 4.0]), {}, 0)

    np.testing.assert_array_equal(action, [3.0, 4.0])
    assert source.follower_source == "hold"
    errors = [event for event in _drain(events) if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert "still inside the model" in errors[0].message


def test_worker_died_is_fatal() -> None:
    source, mailbox, _ = _source()

    class _Policy:
        def set_task(self, task: str | None) -> None:
            return None

        def reset(self, *, reset_model: bool = True) -> None:
            return None

        def update(self, robot_state, camera_frames, step):
            raise WorkerDiedError("Inference thread died")

        def disconnect(self) -> None:
            return None

    source._set_policy(_Policy(), generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    with pytest.raises(WorkerDiedError, match="Inference thread died"):
        source.update(_observation([1.0, 2.0]), {}, 0)


def test_the_model_input_carries_the_task_string() -> None:
    model = FakeInferenceModel(chunk=np.zeros((2, 2), dtype=np.float32), input_names=[STATE, IMAGES])

    policy = PolicySource(
        model=model,
        execution=AsyncExecution(request_threshold=0.5),
        action_queue=ChunkedActionQueue(smoother=LerpSmoother()),
        task=None,
    )
    source, mailbox, _ = _source()
    policy.connect(bus=MagicMock(), session_id="test")
    source._set_policy(policy, generation=1)
    mailbox.apply(StartTaskCommand(task="pick up the red cube"))

    frame = Frame(data=np.zeros((2, 2, 3), dtype=np.uint8), timestamp=1.0, sequence=1)
    source.update(_observation([0.0, 0.0]), {"front": frame}, 0)

    assert model.predict_calls
    assert model.predict_calls[0]["task"] == ["pick up the red cube"]
    policy.disconnect()


def test_the_model_input_is_rgb() -> None:
    model = FakeInferenceModel(chunk=np.zeros((2, 2), dtype=np.float32), input_names=[STATE, IMAGES])

    policy = PolicySource(
        model=model,
        execution=AsyncExecution(request_threshold=0.5),
        action_queue=ChunkedActionQueue(smoother=LerpSmoother()),
        task=None,
    )
    source, mailbox, _ = _source()
    policy.connect(bus=MagicMock(), session_id="test")
    source._set_policy(policy, generation=1)
    mailbox.apply(StartTaskCommand(task="go"))

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    frame = Frame(data=rgb, timestamp=1.0, sequence=1)
    source.update(_observation([0.0, 0.0]), {"front": frame}, 0)

    images = model.predict_calls[0][IMAGES]
    np.testing.assert_array_equal(images[0, 0, 0], [10, 20, 30])
    policy.disconnect()


def test_no_policy_is_not_an_error() -> None:
    follower = FakeRobot([_observation([1.0, 2.0]), _observation([1.0, 2.0])])
    leader = FakeRobot([_observation([4.0, 5.0])])
    source, mailbox, events = _source(follower=follower, leader=leader)

    hold = source.update(follower.get_observation(), {}, 0)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    teleop = source.update(follower.get_observation(), {}, 1)

    np.testing.assert_array_equal(hold, [1.0, 2.0])
    np.testing.assert_array_equal(teleop, [4.0, 5.0])
    assert not any(isinstance(event, ErrorEvent) for event in _drain(events))


def test_start_task_without_a_model_reports_policy_not_loaded() -> None:
    source, mailbox, events = _source()
    mailbox.apply(StartTaskCommand(task="pick"))

    source.update(_observation([1.0, 2.0]), {}, 0)

    assert source.follower_source == "hold"
    errors = [event for event in _drain(events) if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].error_code == "policy_not_loaded"


def test_set_follower_source_policy_is_allowed_when_loaded() -> None:
    source, mailbox, _ = _source()

    class _Policy:
        def set_task(self, task: str | None) -> None:
            return None

        def reset(self, *, reset_model: bool = True) -> None:
            return None

        def update(self, robot_state, camera_frames, step):
            return np.array([9.0, 9.0], dtype=np.float32)

        def disconnect(self) -> None:
            return None

    source._set_policy(_Policy(), generation=1)  # type: ignore[arg-type]
    mailbox.apply(SetFollowerSourceCommand(follower_source="policy"))
    action = source.update(_observation([1.0, 2.0]), {}, 0)

    assert source.follower_source == "policy"
    np.testing.assert_array_equal(action, [9.0, 9.0])

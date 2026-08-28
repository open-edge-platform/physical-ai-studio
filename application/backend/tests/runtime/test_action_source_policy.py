from __future__ import annotations

import threading
import time
from itertools import pairwise
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


def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise TimeoutError("condition was not met")


def _wait_for_policy(source: StudioActionSource) -> None:
    _wait_until(lambda: source.follower_source == "policy")


class _FakePolicy:
    def __init__(
        self,
        *,
        action: list[float] | None = None,
        warmup_delay: float = 0.0,
        warmup_error: Exception | None = None,
        reset_error: Exception | None = None,
        update_error: Exception | None = None,
        order: list[str] | None = None,
        source: StudioActionSource | None = None,
    ) -> None:
        self._action = action
        self._warmup_delay = warmup_delay
        self._warmup_error = warmup_error
        self._reset_error = reset_error
        self._update_error = update_error
        self._order = order
        self._source = source
        self.warmup_threads: list[int] = []
        self.warmup_observations: list[dict] = []

    def set_task(self, task: str | None) -> None:
        if self._order is not None:
            self._order.append(f"set_task:{task}")

    def reset(self, *, reset_model: bool = True) -> None:
        if self._order is not None:
            self._order.append("reset")
        if self._source is not None:
            assert self._source.follower_source != "policy"
        if self._reset_error is not None:
            raise self._reset_error

    def warmup(self, observation: dict) -> None:
        self.warmup_threads.append(threading.get_ident())
        self.warmup_observations.append(observation)
        if self._order is not None:
            self._order.append("warmup")
        if self._warmup_delay > 0:
            time.sleep(self._warmup_delay)
        if self._warmup_error is not None:
            raise self._warmup_error

    def to_model_input(self, robot_state, camera_frames) -> dict:
        return {STATE: np.array([robot_state.joint_positions], dtype=np.float32)}

    def update(self, robot_state, camera_frames, step):
        if self._update_error is not None:
            raise self._update_error
        if self._action is not None:
            return np.array(self._action, dtype=np.float32)
        return np.array(robot_state.joint_positions, dtype=np.float32)

    def disconnect(self) -> None:
        return None


def test_re_arming_resets_before_switching_mode() -> None:
    order: list[str] = []
    source, mailbox, _ = _source()
    policy = _FakePolicy(order=order, source=source)
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    source.update(_observation([1.0, 2.0]), {}, 0)
    _wait_for_policy(source)

    assert order[:2] == ["set_task:pick", "reset"]
    assert "warmup" in order
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
    source.update(_observation([1.0, 2.0]), {}, 0)
    _wait_for_policy(source)
    first = source.update(_observation([1.0, 2.0]), {}, 1)
    np.testing.assert_array_almost_equal(model.predict_calls[0][STATE][0], [1.0, 2.0])
    np.testing.assert_array_almost_equal(first, [1.0, 2.0])

    mailbox.apply(StopTaskCommand())
    source.update(_observation([9.0, 8.0]), {}, 2)
    assert source.follower_source == "hold"

    mailbox.apply(StartTaskCommand(task="go"))
    source.update(_observation([9.0, 8.0]), {}, 3)
    _wait_for_policy(source)
    rearmed = source.update(_observation([9.0, 8.0]), {}, 4)
    np.testing.assert_array_almost_equal(model.predict_calls[-1][STATE][0], [9.0, 8.0])
    assert not np.array_equal(rearmed, [1.0, 2.0])
    policy.disconnect()


def test_play_warmup_runs_off_the_loop_thread() -> None:
    source, mailbox, _ = _source()
    policy = _FakePolicy()
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))
    loop_thread = threading.get_ident()

    source.update(_observation([1.0, 2.0]), {}, 0)
    _wait_for_policy(source)

    assert policy.warmup_threads
    assert all(thread != loop_thread for thread in policy.warmup_threads)
    before = len(policy.warmup_threads)
    source.update(_observation([1.0, 2.0]), {}, 1)
    assert len(policy.warmup_threads) == before


def test_play_warmup_does_not_stall_the_loop() -> None:
    source, mailbox, _ = _source(fps=50)
    policy = _FakePolicy(warmup_delay=0.2)
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    timestamps: list[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < 0.35:
        tick = time.perf_counter()
        source.update(_observation([1.0, 2.0]), {}, len(timestamps))
        timestamps.append(tick)
        remaining = 1 / 50 - (time.perf_counter() - tick)
        if remaining > 0:
            time.sleep(remaining)

    _wait_for_policy(source)
    intervals = [later - earlier for earlier, later in pairwise(timestamps)]
    assert intervals
    assert max(intervals) < 3 / 50
    source.shutdown_policy()


def test_failed_warmup_stays_in_hold() -> None:
    source, mailbox, events = _source()
    policy = _FakePolicy(warmup_error=RuntimeError("compile failed"))
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    source.update(_observation([1.0, 2.0]), {}, 0)
    seen: list = []

    def _has_warmup_error() -> bool:
        seen.extend(_drain(events))
        return any(isinstance(event, ErrorEvent) and event.error_code == "policy_warmup_failed" for event in seen)

    _wait_until(_has_warmup_error)
    assert source.follower_source == "hold"
    hold = source.update(_observation([1.0, 2.0]), {}, 1)
    np.testing.assert_array_equal(hold, [1.0, 2.0])


def test_stop_during_warmup_does_not_switch_to_policy() -> None:
    source, mailbox, _ = _source()
    policy = _FakePolicy(warmup_delay=0.2)
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))
    source.update(_observation([1.0, 2.0]), {}, 0)
    assert source.follower_source == "hold"

    mailbox.apply(StopTaskCommand())
    source.update(_observation([1.0, 2.0]), {}, 1)
    time.sleep(0.3)
    assert source.follower_source == "hold"


def test_a_straggler_error_on_re_arm_keeps_the_session_alive() -> None:
    source, mailbox, events = _source()
    policy = _FakePolicy(
        reset_error=RuntimeError(
            "Previous inference worker is still inside the model after 12.0s. Refusing to start a second worker."
        ),
        action=[99.0, 99.0],
    )
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))

    action = source.update(_observation([3.0, 4.0]), {}, 0)

    np.testing.assert_array_equal(action, [3.0, 4.0])
    assert source.follower_source == "hold"
    errors = [event for event in _drain(events) if isinstance(event, ErrorEvent)]
    assert len(errors) == 1
    assert "still inside the model" in errors[0].message


def test_worker_died_is_fatal() -> None:
    source, mailbox, _ = _source()
    policy = _FakePolicy(update_error=WorkerDiedError("Inference thread died"), warmup_delay=0.05)
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(StartTaskCommand(task="pick"))
    source.update(_observation([1.0, 2.0]), {}, 0)
    _wait_for_policy(source)

    with pytest.raises(WorkerDiedError, match="Inference thread died"):
        source.update(_observation([1.0, 2.0]), {}, 1)


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
    _wait_until(lambda: bool(model.predict_calls))

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
    _wait_until(lambda: bool(model.predict_calls))

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
    policy = _FakePolicy(action=[9.0, 9.0])
    source._set_policy(policy, generation=1)  # type: ignore[arg-type]
    mailbox.apply(SetFollowerSourceCommand(follower_source="policy"))
    source.update(_observation([1.0, 2.0]), {}, 0)
    _wait_for_policy(source)
    action = source.update(_observation([1.0, 2.0]), {}, 1)

    assert source.follower_source == "policy"
    np.testing.assert_array_equal(action, [9.0, 9.0])

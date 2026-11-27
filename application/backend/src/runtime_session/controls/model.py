from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from physicalai.runtime import (
    AsyncExecution,
    ChunkedActionQueue,
    LerpSmoother,
    PolicySource,
    RTCActionQueue,
    RTCExecution,
    SyncExecution,
)
from physicalai.runtime.events import TickEvent

from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE

if TYPE_CHECKING:
    import numpy as np
    from physicalai.runtime.execution.base import Execution
    from physicalai.runtime.execution.queue import ActionQueue

    from runtime_session.model import ProcessModel
    from runtime_session.rerun import RerunObserver, _ModelEvents
    from runtime_session.store import Store
    from schemas.runtime_session import ModelControlConfig


@dataclass
class _Observation:
    """The follower's state as ``PolicySource`` takes it; the model worker adds the frames from the store."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Any] | None = field(default=None)

    @property
    def state(self) -> np.ndarray:
        return self.joint_positions


class ModelControl(Control):
    """Command whatever the loaded model's executor gives for the follower's state, at ``hz``.

    ``physicalai.runtime``'s ``PolicySource`` decides when to ask the model for a
    chunk, queues and smooths it, as an exported runtime does; only the model
    runs in its own process. With ``async`` execution the chunk is requested on the executor's
    own thread while the queue keeps playing. ``rtc`` does too, and guides each new chunk to
    continue the actions still queued; it needs an export with RTC inputs.

    A model can command a pose far from where the follower is (its rest pose,
    say), so for the first ``grace_s`` each action is stamped due at the end of
    it: the robot worker closes the gap over that time rather than jumping. The
    model keeps running on the measured state meanwhile.
    """

    def __init__(
        self, store: Store, model: ProcessModel, config: ModelControlConfig, *, observer: RerunObserver | None = None
    ) -> None:
        if model.inputs.state_size != len(store.spec.joint_names):
            raise ValueError("The loaded model does not fit the follower")
        if model.inputs.tasks and config.task not in model.inputs.tasks:
            raise ValueError(f"Pick one of the model's tasks: {', '.join(model.inputs.tasks)}")
        super().__init__(store, config, name="ModelControl", hz=config.hz)
        self.model = model
        execution, queue = _execution(model, config)
        self._rtc = isinstance(execution, RTCExecution)
        self._queue = queue
        self.source = PolicySource(cast("Any", model), execution, queue, task=config.task or None)
        self.step = 0
        self.grace_s = config.grace_s
        self.started_at = 0.0
        self._observer = observer
        self._events: _ModelEvents | None = None

    def on_start(self) -> None:
        self.started_at = time.monotonic()
        self._events = self._observer.control_events(self.hz, self.grace_s) if self._observer is not None else None
        if self._events is not None and isinstance(self._queue, _RTCQueue):
            self._events.dropped_from = self._queue.dropped_from
        session_id = self._observer.session_id if self._observer is not None else ""
        self.source.connect(bus=cast("Any", self._events), session_id=session_id)

    def on_stop(self) -> None:
        self.source.disconnect()

    def on_tick(self) -> None:
        started = time.monotonic()
        state = self.store.read(STATE)
        if state is None:
            return  # nothing to infer from until the follower is measured
        observation = _Observation(state.value, state.timestamp)
        if self._events is not None:
            self._events.step = self.step
        try:
            action = self.source.update(observation, {}, self.step)
        except Exception as error:
            # RTC runs the model on its own thread and fails with its own summary; say what the model said.
            # Sync and Async report the model's error themselves, so theirs is left as it is.
            if self._rtc and (cause := self.model.last_error) is not None and cause is not error:
                raise cause from error
            raise
        due = max(time.monotonic() + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, action, timestamp=due)
        if self._events is not None:
            duration = time.monotonic() - started
            self._events.emit_tick(
                TickEvent(
                    session_id=self._events.observer.session_id,
                    step=self.step,
                    timestamp=time.time(),
                    robot_state=cast("Any", observation),
                    camera_frames={},  # The observer copies the latest frames off the control thread.
                    action_sent=action.copy(),  # A target; the robot worker owns interpolation.
                    loop_duration_s=duration,
                    sleep_time_s=max(0.0, 1 / self.hz - duration),
                    stale_obs=started - state.timestamp > 2 / self.hz,
                )
            )
        self.step += 1


class _RTCQueue(RTCActionQueue):
    """``RTCActionQueue`` that remembers how many actions of its last chunk it dropped on the merge.

    ``RTCExecution`` reports every chunk with offset 0, though the merge drops the actions played while it was
    inferred; Rerun would plot those as still to come.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last: tuple[np.ndarray, int] | None = None

    def merge(self, raw: np.ndarray, processed: np.ndarray, action_index_before_inference: int | None = None) -> None:
        super().merge(raw, processed, action_index_before_inference)
        # Only RTC's thread merges, and a pop moves the cursor, not the installed chunk.
        with self._lock:
            installed = 0 if self._processed is None else len(self._processed)
        self._last = (processed, len(processed) - installed)

    def dropped_from(self, chunk: np.ndarray) -> int | None:
        """How many leading actions of ``chunk`` the last merge dropped, if ``chunk`` is the one it merged."""
        last = self._last
        return last[1] if last is not None and last[0] is chunk else None


def check_execution(model: ProcessModel, config: ModelControlConfig) -> None:
    """Refuse an execution mode the loaded export cannot run, before anything is stopped for it.

    Raises:
        ValueError: ``rtc`` with an export that takes no RTC inputs, which would run unguided without a word,
            or a horizon the export's chunk has no room for.
    """
    settings = config.execution
    if settings.mode != "rtc":
        return
    if model.inputs.rtc_chunk is None:
        raise ValueError(
            f"Model {model.model.name!r} ({model.backend}) was exported without RTC inputs, so it cannot run with "
            "real-time chunking: choose Synchronous or Asynchronous, or load an RTC export"
        )
    chunk_size = model.inputs.rtc_chunk[0]
    if settings.execution_horizon > chunk_size // 2:
        raise ValueError(f"The RTC horizon is at most half the chunk: {settings.execution_horizon} > {chunk_size // 2}")


def _execution(model: ProcessModel, config: ModelControlConfig) -> tuple[Execution, ActionQueue]:
    """The executor and queue for ``config``'s mode.

    Raises:
        ValueError: See ``check_execution``.
    """
    check_execution(model, config)
    settings = config.execution
    if settings.mode != "rtc":
        cls = AsyncExecution if settings.mode == "async" else SyncExecution
        return cls(request_threshold=settings.request_threshold), ChunkedActionQueue(smoother=LerpSmoother())
    rtc_chunk = cast("tuple[int, int]", model.inputs.rtc_chunk)  # check_execution refused an export without it
    execution = RTCExecution(
        chunk_size=rtc_chunk[0],
        execution_horizon=settings.execution_horizon,
        fps=config.hz,
        max_guidance_weight=settings.max_guidance_weight,
        queue_threshold=None,  # the horizon plus the measured delay
        initial_latency_s=settings.initial_latency_s,
    )
    return execution, _RTCQueue()

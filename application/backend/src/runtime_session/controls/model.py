from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from physicalai.runtime import AsyncExecution, ChunkedActionQueue, LerpSmoother, PolicySource, SyncExecution

from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE

if TYPE_CHECKING:
    import numpy as np

    from runtime_session.model import ProcessModel
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
    own thread while the queue keeps playing.

    A model can command a pose far from where the follower is (its rest pose,
    say), so for the first ``grace_s`` each action is stamped due at the end of
    it: the robot worker closes the gap over that time rather than jumping. The
    model keeps running on the measured state meanwhile.
    """

    def __init__(self, store: Store, model: ProcessModel, config: ModelControlConfig) -> None:
        if model.inputs.state_size != len(store.spec.joint_names):
            raise ValueError("The loaded model does not fit the follower")
        super().__init__(store, config, name="ModelControl", hz=config.hz)
        execution = (
            AsyncExecution(request_threshold=config.execution.request_threshold)
            if config.execution.mode == "async"
            else SyncExecution(request_threshold=config.execution.request_threshold)
        )
        self.source = PolicySource(
            cast("Any", model), execution, ChunkedActionQueue(smoother=LerpSmoother()), task=config.task or None
        )
        self.step = 0
        self.grace_s = config.grace_s
        self.started_at = 0.0

    def on_start(self) -> None:
        self.started_at = time.monotonic()
        # Without callbacks there is no bus to give it; executors only emit to one they have.
        self.source.connect(bus=cast("Any", None), session_id="")

    def on_stop(self) -> None:
        self.source.disconnect()

    def on_tick(self) -> None:
        state = self.store.read(STATE)
        if state is None:
            return  # nothing to infer from until the follower is measured
        action = self.source.update(_Observation(state.value, state.timestamp), {}, self.step)
        self.step += 1
        due = max(time.monotonic() + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, action, timestamp=due)

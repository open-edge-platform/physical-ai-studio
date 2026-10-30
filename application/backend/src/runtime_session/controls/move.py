from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from runtime_session.controls.base import Control
from runtime_session.store import ACTION, STATE

if TYPE_CHECKING:
    from runtime_session.store import Store
    from schemas.runtime_session import MoveControlConfig


class MoveControl(Control):
    """Interpolate from where the follower is to a target over ``duration_s``, then hold the target."""

    def __init__(self, store: Store, config: MoveControlConfig) -> None:
        target = config.position
        unknown = set(target) - set(store.spec.joint_names)
        if unknown:
            raise ValueError(f"Unknown joints {sorted(unknown)}; the follower has {list(store.spec.joint_names)}")
        super().__init__(store, config, name="MoveControl", hz=config.hz)
        self.target_by_name = target
        self.duration_s = config.duration_s
        self.start_position: np.ndarray = np.zeros(len(store.spec.joint_names))
        self.target: np.ndarray = self.start_position
        self.started_at = 0.0

    def on_start(self) -> None:
        state = self.store.read(STATE)
        if state is None:
            raise ValueError("The follower has not been measured yet")
        self.start_position = state.value
        self.target = np.array(
            [self.target_by_name.get(name, start) for name, start in zip(state.names, state.value, strict=True)]
        )
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        now = time.monotonic()
        fraction = min((now - self.started_at) / self.duration_s, 1.0)
        self.store.write(ACTION, self.start_position + (self.target - self.start_position) * fraction, timestamp=now)

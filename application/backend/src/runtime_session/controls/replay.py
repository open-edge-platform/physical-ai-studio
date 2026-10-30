from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from runtime_session.controls.base import Control
from runtime_session.store import ACTION

if TYPE_CHECKING:
    from runtime_session.store import Store
    from schemas.dataset import Episode
    from schemas.runtime_session import ReplayControlConfig


class ReplayControl(Control):
    """Command a recorded episode's actions at the rate it was recorded, then hold its last frame.

    Like teleop, it first commands the episode's first frame due ``grace_s``
    from now, so the robot worker closes the gap over that time rather than
    jumping. Each frame is stamped with when it is due, ``start + grace_s + frame_index / fps``.
    """

    def __init__(self, store: Store, episode: Episode, config: ReplayControlConfig) -> None:
        if tuple(episode.action_keys) != store.spec.joint_names:
            raise ValueError(
                f"Episode {episode.episode_index} drives joints {episode.action_keys},"
                f" the follower has {list(store.spec.joint_names)}"
            )
        if not episode.actions:
            raise ValueError(f"Episode {episode.episode_index} has no frames")
        super().__init__(store, config, name="ReplayControl", hz=episode.fps)
        self.actions = np.asarray(episode.actions, dtype=np.float64)
        self.fps = episode.fps
        self.grace_s = config.grace_s
        self.started_at = 0.0

    def on_start(self) -> None:
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        played_from = self.started_at + self.grace_s
        frame = min(max(int((time.monotonic() - played_from) * self.fps), 0), len(self.actions) - 1)
        self.store.write(ACTION, self.actions[frame], timestamp=played_from + frame / self.fps)

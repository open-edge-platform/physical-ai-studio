from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from robots.features import feature_names
from runtime_session.controls.base import Control
from runtime_session.store import ACTION

if TYPE_CHECKING:
    from physicalai.robot import Robot

    from runtime_session.bus import Bus
    from runtime_session.store import Store
    from schemas.runtime_session import TeleopControlConfig


class TeleopControl(Control):
    """Command the follower whatever the leader measures, due by the leader's next reading.

    The leader can be anywhere when teleop starts, so for the first ``grace_s``
    each command is stamped due at the end of the grace period: the robot worker
    closes the gap over it rather than jumping. It tracks the leader the whole
    time, so the follower is on it when the grace period ends.
    """

    def __init__(self, store: Store, leader: Robot, config: TeleopControlConfig, bus: Bus | None = None) -> None:
        leader_joints = tuple(feature_names(list(leader.joint_names)))
        if leader_joints != store.spec.joint_names:
            raise ValueError(
                f"The leader's joints {leader_joints} do not match the follower's {store.spec.joint_names}"
            )
        super().__init__(store, config, name="TeleopControl", hz=config.hz, bus=bus)
        self.leader = leader
        self.grace_s = config.grace_s
        self.started_at = 0.0

    def on_start(self) -> None:
        # A leader that reads over a connection idle since it connected (a simulation's virtual
        # leader keeps an HTTP connection, which the simulation closes after a few idle seconds)
        # fails its first read, and connects again for the next one.
        try:
            self.leader.get_observation()
        except ConnectionError as error:
            logger.info("The leader's first reading failed ({}); reading it again", error)
            self.leader.get_observation()
        # The grace period starts once the leader reads: a slow first read must not use it up.
        self.started_at = time.monotonic()

    def on_tick(self) -> None:
        observation = self.leader.get_observation()
        due = max(observation.timestamp + 1 / self.hz, self.started_at + self.grace_s)
        self.store.write(ACTION, observation.joint_positions, timestamp=due)

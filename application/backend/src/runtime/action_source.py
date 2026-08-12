from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from runtime.contract import ErrorEvent, SetFollowerSourceCommand, StateData, StateEvent

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import Robot, RobotObservation

    from runtime.contract import CommandMailbox, EventSink, FollowerSource


class StudioActionSource:
    """Select the action sent by RobotRuntime from Studio's current mode."""

    def __init__(
        self,
        *,
        follower: Robot,
        leader: Robot | None,
        mailbox: CommandMailbox,
        event_sink: EventSink,
        fps: float,
    ) -> None:
        self._follower = follower
        self._leader = leader
        self._mailbox = mailbox
        self._event_sink = event_sink
        self._max_leader_failures = max(1, int(3 * fps))
        self._follower_source: FollowerSource = "hold"
        self._hold_target: np.ndarray | None = None
        self._last_leader_action: np.ndarray | None = None
        self._last_leader_timestamp: float | None = None
        self._leader_failures = 0
        self._leader_reads_enabled = leader is not None
        self._failure_logged = False
        self._bus: object | None = None
        self._session_id: str | None = None

    @property
    def follower_source(self) -> FollowerSource:
        return self._follower_source

    def connect(self, *, bus: object, session_id: str) -> None:
        """Connect the leader once and remember runtime context for phase B."""
        self._bus = bus
        self._session_id = session_id
        if self._leader is not None and not self._leader.is_connected():
            self._leader.connect()
        if self._leader is not None and self._leader.joint_names != self._follower.joint_names:
            raise ValueError("Leader and follower joint names must match for teleoperation")

    def update(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],  # noqa: ARG002
        step: int,  # noqa: ARG002
    ) -> np.ndarray:
        self._drain_commands(robot_state)
        if self._hold_target is None:
            self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)

        leader_action = self._read_leader(robot_state) if self._leader_reads_enabled else self._last_leader_action
        policy_action = None

        if self._follower_source == "teleop" and leader_action is not None:
            return leader_action
        if self._follower_source == "policy" and policy_action is not None:
            return policy_action
        return self._hold_target.copy()

    def disconnect(self) -> None:
        """Clear per-run state without disconnecting session-owned devices."""
        self._bus = None
        self._session_id = None
        self._hold_target = None
        self._last_leader_action = None
        self._last_leader_timestamp = None
        self._leader_failures = 0
        self._leader_reads_enabled = self._leader is not None
        self._failure_logged = False

    def _drain_commands(self, robot_state: RobotObservation) -> None:
        for command in self._mailbox.drain():
            if not isinstance(command, SetFollowerSourceCommand):
                continue
            requested = command.follower_source
            if requested == "teleop" and self._leader is None:
                self._event_sink.emit(
                    ErrorEvent(
                        message="Teleoperation requires a leader robot.",
                        error_code="leader_required",
                    )
                )
                continue
            if requested == "teleop" and not self._leader_reads_enabled:
                self._event_sink.emit(
                    ErrorEvent(
                        message="The leader robot is not responding. Reconnect before resuming teleoperation.",
                        error_code="leader_connection_lost",
                    )
                )
                continue
            if requested == "policy":
                self._event_sink.emit(
                    ErrorEvent(
                        message="No policy is loaded.",
                        error_code="policy_not_loaded",
                    )
                )
                continue
            if requested == self._follower_source:
                continue
            self._follower_source = requested
            if requested == "hold":
                self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
            self._emit_state()

    def _read_leader(self, robot_state: RobotObservation) -> np.ndarray | None:
        if self._leader is None:
            return None
        try:
            observation = self._leader.get_observation()
            if self._last_leader_timestamp is not None and observation.timestamp <= self._last_leader_timestamp:
                raise ConnectionError("Leader observation did not advance")
            action = np.array(observation.joint_positions, dtype=np.float32, copy=True)
            if action.shape != robot_state.joint_positions.shape:
                raise ValueError(
                    f"Leader action shape {action.shape} does not match "
                    f"follower shape {robot_state.joint_positions.shape}"
                )
        except Exception as exc:
            self._leader_failures += 1
            if not self._failure_logged:
                logger.warning("Leader observation failed; using the last safe action: {}", exc)
                self._failure_logged = True
            if self._leader_failures >= self._max_leader_failures:
                self._disable_failed_leader(robot_state)
            return self._last_leader_action if self._last_leader_action is not None else self._hold_target

        self._leader_failures = 0
        self._failure_logged = False
        self._last_leader_timestamp = observation.timestamp
        self._last_leader_action = action
        return action

    def _disable_failed_leader(self, robot_state: RobotObservation) -> None:
        self._leader_reads_enabled = False
        self._hold_target = np.array(robot_state.joint_positions, dtype=np.float32, copy=True)
        changed = self._follower_source != "hold"
        self._follower_source = "hold"
        logger.error("Leader failed for {} consecutive ticks; switching to hold", self._leader_failures)
        if changed:
            self._emit_state()
        self._event_sink.emit(
            ErrorEvent(
                message="The leader robot stopped responding. The follower switched to hold.",
                error_code="leader_connection_lost",
            )
        )

    def _emit_state(self) -> None:
        self._event_sink.emit(StateEvent(data=StateData(connected=True, follower_source=self._follower_source)))

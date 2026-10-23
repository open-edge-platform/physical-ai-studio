from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from internal_datasets.mutations.recording_mutation import RecordingMutation
from runtime.features import action_to_dict, observation_to_dict

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.runtime import LifecycleEvent as RuntimeLifecycleEvent
    from physicalai.runtime import TickEvent

    from runtime.contract import FollowerSource

# What the action column of an episode means. ``sent``: the command Studio sent
# on that tick (teleop, policy). ``observed``: Studio sent nothing and another
# controller drove the arm, so the action is the joint positions measured on
# the next tick. One episode never mixes the two.
ActionKind = Literal["sent", "observed"]


class RecordingState:
    """Session-owned recording flags and mutation, shared across threads.

    The runtime is a disposable view; this state outlives a rebuild so an
    episode opened before a reconnect is the same episode after it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dataset_loaded = False
        self._is_recording = False
        self._episodes_recorded = 0
        self._task: str | None = None
        self._mutation: RecordingMutation | None = None
        self._closed = False
        self._episode_action_kind: ActionKind | None = None
        # Observed frames wait one tick for their action (the next measured
        # positions). Only ever holds the latest such frame.
        self._pending_observation: dict[str, Any] | None = None

    @property
    def dataset_loaded(self) -> bool:
        with self._lock:
            return self._dataset_loaded

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @property
    def episodes_recorded(self) -> int:
        with self._lock:
            return self._episodes_recorded

    @property
    def episode_action_kind(self) -> ActionKind | None:
        """Return the action kind the open episode has written, or None if it has written nothing."""
        with self._lock:
            return self._episode_action_kind if self._is_recording else None

    def _end_episode_locked(self) -> None:
        # A buffered observed frame has no next state once the episode ends, so
        # it is dropped rather than written with a made-up action.
        self._pending_observation = None
        self._episode_action_kind = None

    def start(self, task: str) -> bool:
        """Begin an episode. Return False when no dataset is loaded."""
        with self._lock:
            if self._mutation is None or self._closed:
                return False
            self._end_episode_locked()
            self._task = task
            self._is_recording = True
            return True

    def attach_mutation(self, mutation: RecordingMutation) -> None:
        with self._lock:
            self._mutation = mutation
            self._dataset_loaded = True
            self._is_recording = False
            self._end_episode_locked()

    def mark_saved(self) -> None:
        with self._lock:
            self._is_recording = False
            self._end_episode_locked()
            self._episodes_recorded += 1

    def mark_discarded(self) -> None:
        with self._lock:
            self._is_recording = False
            self._end_episode_locked()

    def add_frame(self, observation: dict[str, Any], action: dict[str, float]) -> None:
        """Write one tick under the state lock so save/discard cannot interleave."""
        with self._lock:
            if not self._accepts_locked("sent"):
                return
            if self._mutation is None or self._task is None:
                return
            self._mutation.add_frame(observation, action, self._task)

    def add_observed_frame(self, observation: dict[str, Any], measured_positions: dict[str, float]) -> None:
        """Record a tick on which Studio sent nothing.

        There is no sent action to store, so the action of a frame is the joint
        positions measured on the tick after it. Buffer this tick's frame, and
        write the buffered one with ``measured_positions`` (this tick's state)
        as its action.
        """
        with self._lock:
            if not self._accepts_locked("observed"):
                return
            if self._mutation is None or self._task is None:
                return
            previous = self._pending_observation
            self._pending_observation = observation
            if previous is not None:
                self._mutation.add_frame(previous, measured_positions, self._task)

    def drop_pending_observation(self) -> None:
        """Forget a buffered observed frame whose next tick will not be recorded."""
        with self._lock:
            self._pending_observation = None

    def _accepts_locked(self, kind: ActionKind) -> bool:
        if self._closed or not self._is_recording or self._mutation is None or self._task is None:
            return False
        if self._episode_action_kind is None:
            self._episode_action_kind = kind
        # The action source refuses switches that would mix kinds in an open
        # episode; this guards the tick that races such a switch.
        return self._episode_action_kind == kind

    def stop_episode(self) -> RecordingMutation:
        """Clear the recording flag so ticks skip, then return the mutation.

        Save and discard run off the control thread. Stopping first means an
        in-flight ``add_frame`` finishes (it holds this lock), then later ticks
        see ``is_recording`` is false and skip, then video encode can run.
        """
        with self._lock:
            if not self._is_recording or self._mutation is None:
                raise RuntimeError("No episode is being recorded.")
            self._is_recording = False
            self._end_episode_locked()
            return self._mutation

    def current_mutation(self) -> RecordingMutation | None:
        """Return the attached mutation without requiring an open episode.

        Discard doubles as the recovery path after a failed save, which has
        already cleared the recording flag. Requiring an open episode there
        would leave the buffer with no way to clear it.
        """
        with self._lock:
            self._is_recording = False
            self._end_episode_locked()
            return self._mutation

    def take_mutation(self) -> RecordingMutation | None:
        """Detach the mutation so teardown can finalize it once.

        ``_episodes_recorded`` counts episodes saved into the attached mutation
        and not yet copied into the dataset. Detaching is the moment that count
        becomes zero: the UI adds it to the episodes the dataset API returns, so
        leaving it set double-counts every episode once the copy lands.
        """
        with self._lock:
            mutation = self._mutation
            self._mutation = None
            self._is_recording = False
            self._dataset_loaded = False
            self._task = None
            self._episodes_recorded = 0
            self._end_episode_locked()
            return mutation

    def close(self) -> None:
        with self._lock:
            self._closed = True


class RecordingCallback:
    """Write dataset frames from runtime ticks.

    Kept synchronous: lerobot already runs its own image-writer threads, and
    ordering against ``save_episode`` / ``discard_episode`` has to hold. Do
    not wrap this in ``AsyncCallback``.

    ``close()`` is not optional. ``RobotRuntime.disconnect()`` is permanent
    after the pin bump, Studio never calls it, and ``_CallbackBus.close()``
    therefore never runs. ``RuntimeSession.teardown`` must close this
    explicitly so a later tick cannot write into a finalized mutation.
    """

    def __init__(
        self,
        *,
        recording: RecordingState,
        follower_source: Callable[[], FollowerSource],
    ) -> None:
        self._recording = recording
        self._follower_source = follower_source
        self._joint_names: list[str] = []

    def on_lifecycle(self, event: RuntimeLifecycleEvent) -> None:
        if event.event == "start":
            self._joint_names = list(event.metadata["joint_names"])

    def on_tick(self, event: TickEvent) -> None:
        if not self._joint_names:
            return
        if event.action_sent is None:
            # Observe mode: another controller drove the arm on this tick.
            self._write(
                self._recording.add_observed_frame,
                self._observation(event),
                action_to_dict(self._joint_names, event.robot_state.joint_positions),
            )
            return
        if self._follower_source() == "hold":
            # Hold ticks are not recorded, so a buffered observed frame would
            # otherwise take the state after the gap as its action.
            self._recording.drop_pending_observation()
            return
        self._write(
            self._recording.add_frame,
            self._observation(event),
            action_to_dict(self._joint_names, event.action_sent),
        )

    def _observation(self, event: TickEvent) -> dict[str, Any]:
        observation: dict[str, Any] = dict(observation_to_dict(self._joint_names, event.robot_state))
        for key, frame in event.camera_frames.items():
            observation[key] = np.array(frame.data, copy=True)
        return observation

    @staticmethod
    def _write(
        write: Callable[[dict[str, Any], dict[str, float]], None],
        observation: dict[str, Any],
        action: dict[str, float],
    ) -> None:
        try:
            write(observation, action)
        except Exception:
            logger.exception("Failed to write a recording frame")

    def close(self) -> None:
        self._recording.close()

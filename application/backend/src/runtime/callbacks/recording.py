from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from internal_datasets.mutations.recording_mutation import RecordingMutation
from runtime.features import action_to_dict, observation_to_dict

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.runtime import LifecycleEvent as RuntimeLifecycleEvent
    from physicalai.runtime import TickEvent

    from runtime.contract import FollowerSource


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

    def start(self, task: str) -> bool:
        """Begin an episode. Return False when no dataset is loaded."""
        with self._lock:
            if self._mutation is None or self._closed:
                return False
            self._task = task
            self._is_recording = True
            return True

    def attach_mutation(self, mutation: RecordingMutation) -> None:
        with self._lock:
            self._mutation = mutation
            self._dataset_loaded = True
            self._is_recording = False

    def mark_saved(self) -> None:
        with self._lock:
            self._is_recording = False
            self._episodes_recorded += 1

    def mark_discarded(self) -> None:
        with self._lock:
            self._is_recording = False

    def add_frame(self, observation: dict[str, Any], action: dict[str, float]) -> None:
        """Write one tick under the state lock so save/discard cannot interleave."""
        with self._lock:
            if self._closed or not self._is_recording or self._mutation is None or self._task is None:
                return
            self._mutation.add_frame(observation, action, self._task)

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
        if self._follower_source() == "hold":
            return
        if event.action_sent is None or not self._joint_names:
            return
        observation: dict[str, Any] = dict(
            observation_to_dict(
                self._joint_names,
                event.robot_state,
                include_velocities=False,
            )
        )
        for key, frame in event.camera_frames.items():
            observation[key] = np.array(frame.data, copy=True)
        action = action_to_dict(self._joint_names, event.action_sent)
        try:
            self._recording.add_frame(observation, action)
        except Exception:
            logger.exception("Failed to write a recording frame")

    def close(self) -> None:
        self._recording.close()

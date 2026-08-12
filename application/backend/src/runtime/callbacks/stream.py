from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from runtime.contract import LifecycleData, LifecycleEvent, ObservationEvent, StateData, StateEvent
from runtime.features import observation_to_dict

if TYPE_CHECKING:
    from collections.abc import Callable

    from physicalai.runtime import LifecycleEvent as RuntimeLifecycleEvent
    from physicalai.runtime import TickEvent

    from runtime.contract import EventSink, FollowerSource


class StreamCallback:
    """Translate runtime callbacks into Studio's websocket event contract."""

    def __init__(
        self,
        *,
        event_sink: EventSink,
        follower_source: Callable[[], FollowerSource],
        ready: threading.Event | None = None,
        start_allowed: Callable[[], bool] | None = None,
        lifecycle_lock: threading.Lock | None = None,
    ) -> None:
        self._event_sink = event_sink
        self._follower_source = follower_source
        self._joint_names: list[str] = []
        self.ready = ready or threading.Event()
        self._start_allowed = start_allowed or (lambda: True)
        self._lifecycle_lock = lifecycle_lock or threading.Lock()

    def on_tick(self, event: TickEvent) -> None:
        # Positions only: the browser maps ".pos" features onto the URDF model
        # and drops everything else. Velocities come back with the dataset
        # features in phase B, where a declared schema is what needs them.
        data = observation_to_dict(
            self._joint_names,
            event.robot_state,
            include_velocities=False,
        )
        self._event_sink.emit(ObservationEvent(data=data))

    def on_lifecycle(self, event: RuntimeLifecycleEvent) -> None:
        if event.event == "start":
            with self._lifecycle_lock:
                if not self._start_allowed():
                    return
                self._joint_names = list(event.metadata["joint_names"])
                self.ready.set()
                self._event_sink.emit(
                    StateEvent(
                        data=StateData(
                            connected=True,
                            follower_source=self._follower_source(),
                        )
                    )
                )
        elif event.event == "shutdown":
            self._event_sink.emit(
                LifecycleEvent(
                    data=LifecycleData(
                        event="shutdown",
                        reason=event.metadata.get("reason"),
                        metadata=event.metadata,
                    )
                )
            )

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

_connect_tracker = {"depth": 0, "max_depth": 0}
_connect_depth_lock = threading.Lock()


def reset_connect_tracking() -> None:
    with _connect_depth_lock:
        _connect_tracker["depth"] = 0
        _connect_tracker["max_depth"] = 0


def max_concurrent_connects() -> int:
    with _connect_depth_lock:
        return int(_connect_tracker["max_depth"])


@dataclass
class FakeObservation:
    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict | None = None

    @property
    def state(self) -> np.ndarray:
        if self.sensor_data is None or "velocities" not in self.sensor_data:
            return self.joint_positions
        return np.concatenate((self.joint_positions, self.sensor_data["velocities"]))


class FakeRobot:
    def __init__(
        self,
        observations: list[FakeObservation] | None = None,
        *,
        positions: list[list[float]] | None = None,
        joint_names: list[str] | None = None,
        connect_error: str | None = None,
        observation_error: str | None = None,
        connect_delay: float = 0.0,
        name: str = "fake_robot",
    ) -> None:
        if observations is None:
            observations = [
                FakeObservation(np.array(values, dtype=np.float32), timestamp=float(index + 1))
                for index, values in enumerate(positions or [[0.0]])
            ]
        self._observations = observations
        self._observation_index = 0
        self._connected = False
        self._joint_names = joint_names or [f"joint_{index}" for index in range(len(observations[0].joint_positions))]
        self._connect_error = connect_error
        self._observation_error = observation_error
        self._connect_delay = connect_delay
        self.name = name
        self.sent_actions: list[np.ndarray] = []

    def connect(self) -> None:
        with _connect_depth_lock:
            _connect_tracker["depth"] += 1
            _connect_tracker["max_depth"] = max(_connect_tracker["max_depth"], _connect_tracker["depth"])
        try:
            if self._connect_delay > 0:
                time.sleep(self._connect_delay)
            if self._connect_error is not None:
                raise ConnectionError(self._connect_error)
            self._connected = True
        finally:
            with _connect_depth_lock:
                _connect_tracker["depth"] -= 1

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> FakeObservation:
        if self._observation_error is not None:
            raise ConnectionError(self._observation_error)
        observation = self._observations[min(self._observation_index, len(self._observations) - 1)]
        self._observation_index += 1
        return observation

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        self.sent_actions.append(np.array(action, copy=True))

    def is_connected(self) -> bool:
        return self._connected

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def device_ids(self) -> tuple[str, ...]:
        return ()

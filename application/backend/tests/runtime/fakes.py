from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from physicalai.capture import Frame

_connect_tracker = {"depth": 0, "max_depth": 0}
_connect_depth_lock = threading.Lock()
_disconnect_order: list[str] = []
_disconnect_lock = threading.Lock()


def reset_connect_tracking() -> None:
    with _connect_depth_lock:
        _connect_tracker["depth"] = 0
        _connect_tracker["max_depth"] = 0


def max_concurrent_connects() -> int:
    with _connect_depth_lock:
        return int(_connect_tracker["max_depth"])


def reset_disconnect_tracking() -> None:
    with _disconnect_lock:
        _disconnect_order.clear()


def recorded_disconnects() -> list[str]:
    with _disconnect_lock:
        return list(_disconnect_order)


def _track_connect(delay: float, error: str | None) -> None:
    with _connect_depth_lock:
        _connect_tracker["depth"] += 1
        _connect_tracker["max_depth"] = max(_connect_tracker["max_depth"], _connect_tracker["depth"])
    try:
        if delay > 0:
            time.sleep(delay)
        if error is not None:
            raise ConnectionError(error)
    finally:
        with _connect_depth_lock:
            _connect_tracker["depth"] -= 1


def _track_disconnect(name: str, error: str | None) -> None:
    with _disconnect_lock:
        _disconnect_order.append(name)
    if error is not None:
        raise ConnectionError(error)


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
        disconnect_error: str | None = None,
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
        self._disconnect_error = disconnect_error
        self.name = name
        self.sent_actions: list[np.ndarray] = []

    def connect(self) -> None:
        _track_connect(self._connect_delay, self._connect_error)
        self._connected = True

    def disconnect(self) -> None:
        _track_disconnect(self.name, self._disconnect_error)
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


class FakeAdapter:
    def __init__(self, input_names: list[str]) -> None:
        self.input_names = input_names


class FakeInferenceModel:
    """Stand-in for physicalai.inference.InferenceModel used by PolicyLoader tests."""

    def __init__(
        self,
        export_dir: object = None,
        policy_name: str | None = None,
        backend: str = "auto",
        device: str = "auto",
        *,
        input_names: list[str] | None = None,
        chunk: np.ndarray | None = None,
        construct_delay: float = 0.0,
        label: str = "fake",
        predict: object | None = None,
    ) -> None:
        if construct_delay > 0:
            time.sleep(construct_delay)
        self.export_dir = export_dir
        self.policy_name = policy_name
        self.backend = backend
        self.device = device
        self.adapter = FakeAdapter(input_names or ["state"])
        self._chunk = np.zeros((4, 1), dtype=np.float32) if chunk is None else np.asarray(chunk, dtype=np.float32)
        self.chunk_size = int(self._chunk.shape[0])
        self.predict_calls: list[dict] = []
        self.reset_calls = 0
        self.label = label
        self._predict = predict

    def predict_action_chunk(self, observation: dict) -> np.ndarray:
        self.predict_calls.append(observation)
        if callable(self._predict):
            return np.asarray(self._predict(observation), dtype=np.float32)
        return np.array(self._chunk, copy=True)

    def reset(self) -> None:
        self.reset_calls += 1


class FakeCamera:
    """Stand-in for SharedCamera that participates in session connect/teardown."""

    def __init__(
        self,
        *,
        name: str = "fake_camera",
        connect_error: str | None = None,
        disconnect_error: str | None = None,
        connect_delay: float = 0.0,
        width: int = 8,
        height: int = 8,
    ) -> None:
        self.name = name
        self._connect_error = connect_error
        self._disconnect_error = disconnect_error
        self._connect_delay = connect_delay
        self._width = width
        self._height = height
        self._connected = False
        self._sequence = 0

    def connect(self, timeout: float = 5.0) -> None:
        _track_connect(self._connect_delay, self._connect_error)
        self._connected = True

    def disconnect(self) -> None:
        _track_disconnect(self.name, self._disconnect_error)
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_latest(self) -> Frame:
        self._sequence += 1
        return Frame(
            data=np.zeros((self._height, self._width, 3), dtype=np.uint8),
            timestamp=time.monotonic(),
            sequence=self._sequence,
        )

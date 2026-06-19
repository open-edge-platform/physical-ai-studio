from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraRegistryEntry:
    id: str
    name: str
    width: int
    height: int
    frame_data: Any  # mp.Array[c_uint8], shared with CameraWorker


@dataclass
class RobotRegistryEntry:
    name: str
    type: str
    action_read_state: Any  # mp.Value[c_int], shared with TeleoperateWorker
    features: list[str]
    state: Any  # mp.Array[c_double], shared with TeleoperateWorker
    actions: Any  # mp.Array[c_double], shared with TeleoperateWorker


@dataclass
class EnvironmentDataRegistry:
    """Describes all data streams produced by a loaded environment."""

    robot: RobotRegistryEntry
    cameras: list[CameraRegistryEntry] = field(default_factory=list)

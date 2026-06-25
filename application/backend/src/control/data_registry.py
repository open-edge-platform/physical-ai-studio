from dataclasses import dataclass, field
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray


@dataclass
class CameraRegistryEntry:
    id: str
    name: str
    width: int
    height: int
    frame_data: SynchronizedArray  # mp.Array[c_uint8], shared with CameraWorker


@dataclass
class RobotRegistryEntry:
    name: str
    type: str
    action_read_state: Synchronized  # mp.Value[c_int], shared with TeleoperateWorker
    features: list[str]
    state: SynchronizedArray  # mp.Array[c_double], shared with TeleoperateWorker
    actions: SynchronizedArray  # mp.Array[c_double], shared with TeleoperateWorker


@dataclass
class EnvironmentDataRegistry:
    """Describes all data streams produced by a loaded environment."""

    robot: RobotRegistryEntry
    cameras: list[CameraRegistryEntry] = field(default_factory=list)

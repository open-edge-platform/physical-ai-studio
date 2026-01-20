from pydantic import BaseModel

from .camera import CameraConfig
from .dataset import Dataset
from .model import Model
from .robot import RobotConfig


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset
    fps: int
    cameras: list[CameraConfig]
    follower: RobotConfig
    leader: RobotConfig


class InferenceConfig(BaseModel):
    model: Model
    task_index: int
    fps: int
    cameras: list[CameraConfig]
    robot: RobotConfig
    backend: str

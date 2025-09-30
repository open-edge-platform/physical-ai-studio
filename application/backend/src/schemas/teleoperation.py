from pydantic import BaseModel

from .camera import CameraConfig
from .dataset import Dataset
from .robot import RobotConfig


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset | None
    fps: int
    cameras: list[CameraConfig]
    follower: RobotConfig
    leader: RobotConfig

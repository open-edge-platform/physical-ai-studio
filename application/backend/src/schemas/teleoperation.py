from pydantic import BaseModel

from .camera import CameraConfig
from .robot import RobotConfig
from .dataset import Dataset


class TeleoperationConfig(BaseModel):
    task: str
    dataset: Dataset | None
    fps: int
    cameras: list[CameraConfig]
    follower: RobotConfig
    leader: RobotConfig

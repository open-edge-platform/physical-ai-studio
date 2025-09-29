#!/usr/bin/env python3


from pydantic import BaseModel

from .camera import CameraConfig
from .robot import RobotConfig


class TeleoperationConfig(BaseModel):
    task: str
    dataset_id: str | None
    cameras: list[CameraConfig]
    robots: list[RobotConfig]

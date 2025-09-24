#!/usr/bin/env python3

from typing import Literal

from pydantic import BaseModel, Field

from .camera import CameraConfig
from .robot import RobotConfig


class TeleoperationConfig(BaseModel):
    task: str
    dataset_id: str | None
    cameras: list[CameraConfig]
    robots: list[RobotConfig]

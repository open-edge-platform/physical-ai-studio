from typing import Literal

from pydantic import BaseModel


class CalibrationConfig(BaseModel):
    id: str
    path: str
    robot_type: Literal["teleoperator", "robot"]

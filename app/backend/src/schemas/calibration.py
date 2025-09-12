
from pydantic import BaseModel, Field

from typing import Literal

class CalibrationConfig(BaseModel):
    id: str
    path: str
    robot_type: Literal["teleoperator", "robot"]
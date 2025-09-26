from typing import Literal
from pydantic import BaseModel, Field


class RobotPortInfo(BaseModel):
    port: str
    serial_id: str
    robot_type: str


class RobotConfig(BaseModel):
    id: str = Field(description="Robot calibration id")
    type: Literal["follower", "leader"]
    port: str = Field(description="Serial port of robot")
    serial_id: str = Field(description="Serial ID of device")
    robot_type: str = Field(description="Robot Type (e.g. so101)")

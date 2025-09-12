from typing import Literal

from pydantic import BaseModel, Field


class RobotConfig(BaseModel):
    id: str = Field(description="Robot calibration id")
    type: Literal["follower", "leader"]
    serial_id: str = Field(description="Serial port id")

class RobotPortInfo(BaseModel):
    port: str
    serial_id: str
    device_name: str

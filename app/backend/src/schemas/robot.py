from pydantic import BaseModel, Field
from typing import List,Literal

class RobotConfig(BaseModel):
    id: str = Field(None, description="Robot calibration id")
    type: Literal["follower", "leader"]
    serial_id: str = Field(None, description="Serial port id")

class RobotPortInfo(BaseModel):
    port: str
    serial_id: str
    device_name: str

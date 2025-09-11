from pydantic import BaseModel, Field
from typing import List,Literal

#            "follower": {
#                "serial_id": "5AA9017083",
#                "id": "khaos"
#            },

class RobotConfig(BaseModel):
    id: str = Field(None, description="Robot calibration id")
    type: Literal["follower", "leader"]
    serial_id: str = Field(None, description="Serial port id")

from abc import ABC
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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


class RobotType(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"



class Robot(ABC, BaseModel):
    id: Annotated[UUID, Field(description="Unique identifier")]

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    serial_id: str = Field(..., description="Unique serial identifier for the robot")
    type: RobotType = Field(..., description="Type of robot configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "Assembly Line Robot 1",
                "serial_id": "SO101-2024-001",
                "robot_type": "SO101_Leader",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }
    )

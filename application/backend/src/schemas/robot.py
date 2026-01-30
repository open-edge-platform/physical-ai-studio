from abc import ABC
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SerialPortInfo(BaseModel):
    connection_string: str
    serial_number: str
    robot_type: str


class BaseRobotConfig(BaseModel):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type")


class LeRobotConfig(BaseRobotConfig):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type (e.g. so101)")
    id: str = Field(description="Robot calibration id")
    port: str = Field(description="Serial port of robot")
    serial_number: str = Field(description="Serial ID of device")


class NetworkIpRobotConfig(BaseRobotConfig):
    type: Literal["follower", "leader"]
    robot_type: str = Field(description="Robot Type (e.g. Trossen WidowX AI)")
    connection_string: str = Field(description="IP address of robot")


class RobotType(StrEnum):
    SO101_FOLLOWER = "SO101_Follower"
    SO101_LEADER = "SO101_Leader"
    TROSSEN_WIDOWXAI_LEADER = "Trossen_WidowXAI_Leader"
    TROSSEN_WIDOWXAI_FOLLOWER = "Trossen_WidowXAI_Follower"


class Robot(ABC, BaseModel):
    id: Annotated[UUID, Field(description="Unique identifier")]

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    connection_string: str = Field(
        ..., description="Connection string to device, keep empty to automatically find using serial number"
    )
    serial_number: str = Field(..., description="Unique serial number for the robot")
    type: RobotType = Field(..., description="Type of robot configuration")
    active_calibration_id: UUID | None = Field(default=None, description="The ID of the active calibration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "Assembly Line Robot 1",
                "connection_string": "SO101-2024-001",
                "robot_type": "SO101_Leader",
                "active_calibration_id": "b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )

class RobotWithConnectionState(Robot):
    connection_status: Literal["online", "offline", "unknown"] = "unknown"

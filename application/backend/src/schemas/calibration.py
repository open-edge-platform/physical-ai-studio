from abc import ABC
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CalibrationConfig(BaseModel):
    id: str
    path: str
    robot_type: Literal["follower", "leader"]


class CalibrationValue(ABC, BaseModel):
    id: int = Field(..., description="motor id that this joint applies to")
    joint_name: str = Field(..., description="Name of the joint this value applies to")
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "joint_name": "shoulder_pan",
                "drive_mode": 1,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
        }
    )


class Calibration(ABC, BaseModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    robot_id: Annotated[UUID, Field(description="Unique identifier")]

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    file_path: str = Field(..., description="File path to calibration file on disk, used by lerobot")
    values: dict[str, CalibrationValue]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "robot_id": "b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "file_path": "/home/user/robots/b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f/calibrations/a5e2cde6-936b-4a9e-a213-08dda0afa453.json",  # noqa: E501
                "values": {
                    "shoulder_pan": {
                        "id": 1,
                        "joint_name": "shoulder_pan",
                        "drive_mode": 1,
                        "homing_offset": 0,
                        "range_min": 0,
                        "range_max": 4095,
                    }
                },
            }
        }
    )

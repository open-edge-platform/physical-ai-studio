from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from schemas.project_camera import Camera
from schemas.robot import Robot


class TeleoperatorRobot(BaseModel):
    type: Literal["robot"] = "robot"
    robot_id: UUID = Field(..., description="ID of the robot acting as teleoperator")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "robot",
                "robot_id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
            }
        }
    )


class TeleoperatorNone(BaseModel):
    type: Literal["none"] = "none"

    model_config = ConfigDict(json_schema_extra={"example": {"type": "none"}})


Teleoperator = Annotated[TeleoperatorRobot | TeleoperatorNone, Field(discriminator="type")]


class RobotEnvironmentConfiguration(BaseModel):
    robot_id: UUID = Field(..., description="ID of the robot in this environment")
    tele_operator: Teleoperator = Field(..., description="Teleoperator configuration for this robot")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "robot_id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "tele_operator": {
                    "type": "robot",
                    "robot_id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                },
            }
        }
    )


class Environment(BaseModel):
    id: Annotated[UUID, Field(description="Unique identifier")]

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable environment name")
    robots: list[RobotEnvironmentConfiguration] = Field(
        default_factory=list,
        description="List of robot configurations in this environment",
    )
    camera_ids: list[UUID] = Field(default_factory=list, description="List of camera IDs in this environment")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "e7d4bef8-158d-6c1g-c435-20ffb2chc675",
                "name": "Assembly Line Environment",
                "robots": [
                    {
                        "robot_id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                        "tele_operator": {
                            "type": "robot",
                            "robot_id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                        },
                    },
                    {
                        "robot_id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                        "tele_operator": {"type": "none"},
                    },
                ],
                "camera_ids": [
                    "c8g5dfh9-269e-7d2h-d546-31ggc3dic786",
                    "d9h6egi0-370f-8e3i-e657-42hhd4ejd897",
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }
    )


class TeleoperatorRobotWithRobot(BaseModel):
    """Teleoperator configuration with eager-loaded robot."""

    type: Literal["robot"] = "robot"
    robot_id: UUID = Field(..., description="ID of the robot acting as teleoperator")
    robot: Robot | None = Field(None, description="Eager-loaded robot object")


class TeleoperatorNoneWithRobot(BaseModel):
    """Teleoperator configuration for no teleoperator."""

    type: Literal["none"] = "none"


TeleoperatorWithRobot = Annotated[TeleoperatorRobotWithRobot | TeleoperatorNoneWithRobot, Field(discriminator="type")]


class RobotWithTeleoperator(BaseModel):
    """Robot configuration with eager-loaded robot and teleoperator."""

    robot: Robot = Field(..., description="The robot in this environment")
    tele_operator: TeleoperatorWithRobot = Field(..., description="Teleoperator configuration with eager-loaded data")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "robot": {
                    "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                    "name": "Assembly Line Robot 1",
                    "serial_id": "SO101-2024-001",
                    "type": "SO101_Leader",
                    "cameras": [],
                },
                "tele_operator": {
                    "type": "robot",
                    "robot_id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                    "robot": {
                        "id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                        "name": "Teleoperator Robot",
                        "serial_id": "SO101-2024-002",
                        "type": "SO101_Follower",
                        "cameras": [],
                    },
                },
            }
        }
    )


class EnvironmentWithRelations(BaseModel):
    """Environment with eager-loaded robots and cameras."""

    id: Annotated[UUID, Field(description="Unique identifier")]

    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable environment name")
    robots: list[RobotWithTeleoperator] = Field(
        default_factory=list,
        description="List of robots with eager-loaded teleoperators",
    )
    cameras: list[Camera] = Field(default_factory=list, description="List of eager-loaded cameras")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "e7d4bef8-158d-6c1g-c435-20ffb2chc675",
                "name": "Assembly Line Environment",
                "robots": [
                    {
                        "robot": {
                            "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                            "name": "Assembly Line Robot 1",
                            "serial_id": "SO101-2024-001",
                            "type": "SO101_Leader",
                        },
                        "tele_operator": {
                            "type": "robot",
                            "robot_id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                            "robot": {
                                "id": "b6f3def7-047c-5b0f-b324-19eeb1bgb564",
                                "name": "Teleoperator Robot",
                                "serial_id": "SO101-2024-002",
                                "type": "SO101_Follower",
                            },
                        },
                    }
                ],
                "cameras": [
                    {
                        "id": "c8g5dfh9-269e-7d2h-d546-31ggc3dic786",
                        "name": "front_camera",
                        "fingerprint": "/dev/video0",
                        "driver": "webcam",
                        "hardware_name": "Camera 1",
                        "resolution_width": 480,
                        "resolution_height": 360,
                        "resolution_fps": 30,
                    }
                ],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }
    )

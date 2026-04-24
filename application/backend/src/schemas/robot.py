from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from schemas.base import BaseIDModel


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


# ============================================================================
# Payload Models (Configuration Only)
# ============================================================================


class SO101RobotPayload(BaseModel):
    """Connection configuration for SO-101 serial robots."""

    connection_string: str = Field(
        default="",
        description="Serial port path; leave empty to auto-discover via serial_number",
    )
    serial_number: str = Field(..., description="Unique serial number for the robot")


class TrossenSingleArmPayload(BaseModel):
    """Connection configuration for Trossen single-arm robots."""

    connection_string: str = Field(..., description="IP address of the robot")
    serial_number: str = Field(default="", description="Serial number (unused for IP robots)")


# ============================================================================
# Concrete Robot Models
# ============================================================================


_SO101Types = Literal[RobotType.SO101_FOLLOWER, RobotType.SO101_LEADER]
_TrossenTypes = Literal[RobotType.TROSSEN_WIDOWXAI_LEADER, RobotType.TROSSEN_WIDOWXAI_FOLLOWER]


class BaseRobot(BaseIDModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")
    active_calibration_id: UUID | None = Field(default=None, description="The ID of the active calibration")


class SO101Robot(BaseRobot):
    """SO-101 follower or leader robot using a serial connection."""

    type: _SO101Types = Field(..., description="Type of robot configuration")
    payload: SO101RobotPayload = Field(..., description="SO-101 connection configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "Assembly Line Robot 1",
                "type": "SO101_Follower",
                "payload": {
                    "connection_string": "",
                    "serial_number": "SO101-2024-001",
                },
                "active_calibration_id": "b7f3d9e2-1a2b-4c3d-8e9f-0a1b2c3d4e5f",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )


class TrossenSingleArmRobot(BaseRobot):
    """Trossen WidowX AI follower or leader robot using an IP connection."""

    type: _TrossenTypes = Field(..., description="Type of robot configuration")
    payload: TrossenSingleArmPayload = Field(..., description="Trossen single-arm connection configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a5e2cde6-936b-4a9e-a213-08dda0afa453",
                "name": "WidowX AI Robot 1",
                "type": "Trossen_WidowXAI_Follower",
                "payload": {
                    "connection_string": "192.168.1.100",
                    "serial_number": "",
                },
                "active_calibration_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            },
        },
    )


# Discriminated union of all robot types
Robot = Annotated[
    SO101Robot | TrossenSingleArmRobot,
    Field(discriminator="type"),
]

RobotAdapter: TypeAdapter[Robot] = TypeAdapter(Robot)


# ============================================================================
# RobotWithConnectionState variants
# ============================================================================

_ConnectionStatus = Literal["online", "offline", "unknown"]


class SO101RobotWithConnectionState(SO101Robot):
    connection_status: _ConnectionStatus = "unknown"


class TrossenSingleArmRobotWithConnectionState(TrossenSingleArmRobot):
    connection_status: _ConnectionStatus = "unknown"


RobotWithConnectionState = Annotated[
    SO101RobotWithConnectionState | TrossenSingleArmRobotWithConnectionState,
    Field(discriminator="type"),
]

RobotWithConnectionStateAdapter: TypeAdapter[RobotWithConnectionState] = TypeAdapter(RobotWithConnectionState)

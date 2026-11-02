from typing import Annotated
from uuid import UUID

from pydantic import Field

from schemas.base import BaseIDModel, UTCDatetime

RobotType = str


class BaseRobot(BaseIDModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    created_at: UTCDatetime | None = Field(None)
    updated_at: UTCDatetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")

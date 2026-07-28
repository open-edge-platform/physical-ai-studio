from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import Field

from schemas.base import BaseIDModel

RobotType = str


class BaseRobot(BaseIDModel):
    id: Annotated[UUID, Field(description="Unique identifier")]
    created_at: datetime | None = Field(None)
    updated_at: datetime | None = Field(None)

    name: str = Field(..., description="Human-readable robot name")

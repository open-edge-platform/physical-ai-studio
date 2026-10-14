from abc import ABC
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import AfterValidator, BaseModel, Field, field_serializer


def _ensure_timezone(value: datetime) -> datetime:
    """Treat naive datetimes (e.g. read back from SQLite) as UTC.

    Keeps serialized output RFC 3339-compliant (with an offset), as the OpenAPI
    `date-time` format requires.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_timezone)]


class BaseIDModel(ABC, BaseModel):
    """Base model with an id field."""

    id: UUID = Field(default_factory=uuid4)

    @field_serializer("id")
    def serialize_id(self, id: UUID, _info: Any) -> str:
        return str(id)


class BaseIDNameModel(ABC, BaseModel):
    """Base model with id and name fields."""

    id: Annotated[UUID, Field(description="Unique identifier")]
    name: str = "Default Name"


class Pagination(ABC, BaseModel):
    """Pagination model."""

    offset: int  # index of the first item returned (0-based)
    limit: int  # number of items requested per page
    count: int  # number of items actually returned (may be less than limit if at the end)
    total: int  # total number of items available

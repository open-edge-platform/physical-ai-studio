from abc import ABC
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer


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

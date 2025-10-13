from abc import ABC
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field


class BaseIDModel(ABC, BaseModel):
    """Base model with an id field."""

    id: Annotated[UUID, Field(description="Unique identifier")]


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

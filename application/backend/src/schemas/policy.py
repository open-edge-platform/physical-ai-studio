"""Policy references describe availability without changing saved policy identifiers."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AvailablePolicyReference(BaseModel):
    """A policy definition present on the described training target."""

    id: str
    status: Literal["available"] = "available"


class UnavailablePolicy(BaseModel):
    """A referenced policy whose definition or declared dependencies are missing."""

    id: str
    status: Literal["unavailable"] = "unavailable"
    unavailable: Literal[True] = True
    reason: str


class UnknownPolicyAvailability(BaseModel):
    """A policy whose target inventory has not been verified."""

    id: str
    status: Literal["unknown"] = "unknown"
    reason: str


ReadablePolicy = Annotated[
    AvailablePolicyReference | UnavailablePolicy | UnknownPolicyAvailability,
    Field(discriminator="status"),
]

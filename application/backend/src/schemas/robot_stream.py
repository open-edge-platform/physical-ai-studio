from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ObservationEvent(BaseModel):
    event: Literal["observation"] = "observation"
    data: dict[str, float]
    actions: dict[str, float] | None = None


class ActionCommand(BaseModel):
    """A client's joint targets by feature name; joints it leaves out keep their current target."""

    event: Literal["action"]
    data: dict[str, float]


class ErrorEvent(BaseModel):
    event: Literal["error"] = "error"
    message: str
    error_code: str

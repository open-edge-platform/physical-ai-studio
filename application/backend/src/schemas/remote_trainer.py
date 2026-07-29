from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from schemas.hardware import DeviceInfo, StorageInfo

HealthStatus = Literal["healthy", "degraded", "unreachable"]


class RemoteTrainerCreate(BaseModel):
    """Configuration for a direct remote trainer endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    url: AnyHttpUrl


class RemoteTrainerUpdate(BaseModel):
    """Mutable fields for a direct remote trainer endpoint."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    url: AnyHttpUrl | None = None


class RemoteTrainer(RemoteTrainerCreate):
    """Persisted direct remote trainer endpoint."""

    id: UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RemoteTrainerHealth(BaseModel):
    """A sanitized, point-in-time health result for a configured trainer."""

    remote_trainer_id: UUID
    status: HealthStatus
    checked_at: datetime
    latency_ms: int | None = Field(default=None, ge=0)
    devices: list[DeviceInfo] = Field(default_factory=list)
    storage: StorageInfo | None = Field(
        default=None,
        description="Available storage on the trainer, when reported. Absence does not affect health status.",
    )
    reason_code: str | None = None

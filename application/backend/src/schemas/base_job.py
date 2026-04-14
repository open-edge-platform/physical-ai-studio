from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, field_serializer

from schemas.base import BaseIDModel


class JobType(StrEnum):
    TRAINING = "training"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class BaseJob(BaseIDModel):
    project_id: UUID
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage from 0 to 100")
    status: JobStatus = JobStatus.PENDING
    # Optional telemetry/debug details only.
    # Contract: data in `extra_info` must never be required for workflow decisions.
    # If deleting this field would break execution or UI behavior, that data belongs in typed payload.
    extra_info: dict | None = None
    message: str = "Job created"
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime | None = Field(None)

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

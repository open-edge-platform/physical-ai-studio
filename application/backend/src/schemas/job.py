from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

from schemas.base import BaseIDModel


class JobType(StrEnum):
    TRAINING = "training"
    OPTIMIZATION = "optimization"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class Job(BaseIDModel):
    project_id: UUID
    type: JobType = JobType.TRAINING
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage from 0 to 100")
    status: JobStatus = JobStatus.PENDING
    payload: dict
    message: str = "Job created"
    start_time: datetime | None = None
    end_time: datetime | None = None

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)


class JobList(BaseModel):
    jobs: list[Job]


class TrainJobPayload(BaseModel):
    project_id: UUID
    dataset_id: UUID
    policy: str
    model_name: str

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("dataset_id")
    def serialize_dataset_id(self, dataset_id: UUID, _info: Any) -> str:
        return str(dataset_id)

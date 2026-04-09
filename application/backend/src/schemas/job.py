from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

from schemas.base import BaseIDModel


class JobType(StrEnum):
    TRAINING = "training"


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
    extra_info: dict | None = None
    message: str = "Job created"
    start_time: datetime | None = None
    end_time: datetime | None = None
    created_at: datetime | None = Field(None)

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
    max_steps: int = Field(default=100, ge=100, le=100_000, description="Number of training steps")
    batch_size: int = Field(default=8, ge=1, le=256, description="Training batch size")
    num_workers: int | Literal["auto"] = Field(default="auto", description="DataLoader workers ('auto' or 0-16)")
    auto_scale_batch_size: bool = Field(
        default=False, description="Run batch-size finder before training (power scaling)"
    )
    base_model_id: UUID | None = Field(default=None, description="Model ID to resume training from")
    val_split: float = Field(
        default=0.1, ge=0.0, lt=1.0, description="Fraction of episodes to hold out for eval-loss validation (0 = disabled)"
    )

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("dataset_id")
    def serialize_dataset_id(self, dataset_id: UUID, _info: Any) -> str:
        return str(dataset_id)

    @field_serializer("base_model_id")
    def serialize_base_model_id(self, base_model_id: UUID | None, _info: Any) -> str | None:
        return str(base_model_id) if base_model_id else None

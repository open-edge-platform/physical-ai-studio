from typing import Annotated, Any, Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, Field, field_serializer, model_validator

from schemas.base_job import BaseJob, JobType
from schemas.hardware import DeviceType


class JobList(BaseModel):
    jobs: list["Job"]


class TrainingDevice(BaseModel):
    """Device specification for training."""

    type: DeviceType = Field(..., description="Device type, e.g. 'cpu', 'xpu', 'cuda'")
    index: int | None = Field(default=None, ge=0, description="Device index (null for CPU/NPU)")

    @model_validator(mode="after")
    def validate_index_for_device_type(self) -> "TrainingDevice":
        """Ensure index is consistent with the device type.

        Indexed types (cuda, xpu) default to index 0 when omitted.
        Non-indexed types (cpu, npu) ignore a supplied index with a warning.
        """
        device_type_str = str(self.type).lower()
        indexed_types = {"cuda", "xpu"}
        non_indexed_types = {"cpu", "npu"}

        if device_type_str in non_indexed_types:
            if self.index is not None:
                logger.warning(
                    "Device type '{}' does not support an index. Got index={}. Disregarding index.",
                    self.type,
                    self.index,
                )
                self.index = None
        elif device_type_str in indexed_types and self.index is None:
            logger.warning(
                "Device type '{}' requires an index (e.g., 'cuda:0', 'xpu:0'). Using default index 0.",
                device_type_str,
            )
            self.index = 0
        return self


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
    device: TrainingDevice | None = Field(default=None, description="Target training device (auto-detected if null)")

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("dataset_id")
    def serialize_dataset_id(self, dataset_id: UUID, _info: Any) -> str:
        return str(dataset_id)

    @field_serializer("base_model_id")
    def serialize_base_model_id(self, base_model_id: UUID | None, _info: Any) -> str | None:
        return str(base_model_id) if base_model_id else None


class TrainJob(BaseJob):
    type: Literal[JobType.TRAINING] = JobType.TRAINING  # type: ignore[valid-type]
    payload: TrainJobPayload


JobPayload = TrainJobPayload

Job = Annotated[
    TrainJob,
    Field(discriminator="type"),
]

JobList.model_rebuild()

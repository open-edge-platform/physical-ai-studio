from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, Field, field_serializer, model_validator

from schemas.base_job import BaseJob, JobType
from schemas.dataset_import_job import DatasetImportJobPayload
from schemas.hardware import DeviceType


class TrainingPrecision(StrEnum):
    """Supported training precision modes.

    Values align with Lightning's precision strings and can be passed
    directly to ``Trainer(precision=...)``.
    """

    FP32 = "32-true"
    BF16_MIXED = "bf16-mixed"
    BF16_TRUE = "bf16-true"


class TrainingTarget(StrEnum):
    """Where a training job executes."""

    LOCAL = "local"
    REMOTE = "remote"


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


_DEFAULT_MAX_EPOCHS = 5


class TrainJobPayload(BaseModel):
    project_id: UUID
    dataset_id: UUID
    policy: str
    model_name: str
    max_epochs: int | None = Field(
        default=None,
        ge=1,
        le=10_000,
        description="Number of training epochs (preferred over max_steps when both are provided)",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Number of training steps (legacy; ignored when max_epochs is provided)",
    )
    batch_size: int = Field(default=8, ge=1, le=256, description="Training batch size")

    @model_validator(mode="after")
    def resolve_training_limit(self) -> "TrainJobPayload":
        """Resolve training-limit fields, applying precedence and defaults.

        Rules (in order of priority):
        1. If ``max_epochs`` is set, use it (``max_steps`` is ignored).
        2. If ``max_epochs`` is not set (including legacy ``max_steps``-only payloads),
           default to ``_DEFAULT_MAX_EPOCHS`` epochs.

        ``max_steps`` is retained only for backward-compatible payload parsing.
        """
        if self.max_epochs is not None:
            # max_epochs wins; clear max_steps to avoid ambiguity downstream
            self.max_steps = None
        else:
            # No epoch value provided (including legacy max_steps-only payloads)
            self.max_epochs = _DEFAULT_MAX_EPOCHS
        return self

    num_workers: int | Literal["auto"] = Field(default="auto", description="DataLoader workers ('auto' or 0-16)")
    auto_scale_batch_size: bool = Field(
        default=False,
        description="Run batch-size finder before training (power scaling)",
    )
    base_model_id: UUID | None = Field(default=None, description="Model ID to resume training from")
    val_split: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Fraction of episodes to hold out for eval-loss validation (0 = disabled)",
    )
    device: TrainingDevice | None = Field(default=None, description="Target training device (auto-detected if null)")
    precision: TrainingPrecision = Field(
        default=TrainingPrecision.BF16_MIXED,
        description="Training precision ('32-true', 'bf16-mixed')",
    )
    compile_model: bool = Field(default=False, description="Enable torch.compile for supported policies")
    training_target: TrainingTarget = Field(
        default=TrainingTarget.LOCAL,
        description="Whether training runs in the Studio process or on a configured remote trainer",
    )
    remote_trainer_id: UUID | None = Field(
        default=None,
        description="Configured remote trainer selected for a remote run",
    )
    remote_trainer_url: str | None = Field(
        default=None,
        description="Resolved remote trainer URL pinned when the job is submitted for restart recovery",
    )
    remote_trainer_name: str | None = Field(
        default=None,
        description="Resolved remote trainer name pinned when the job is submitted, for display in job logs",
    )

    remote_job_id: UUID | None = Field(
        default=None, description="Remote trainer job id, set when a remote run is in flight (for restart reattach)"
    )
    snapshot_id: UUID | None = Field(
        default=None, description="Dataset snapshot id retained while a remote run is in flight (for model provenance)"
    )

    @model_validator(mode="after")
    def validate_training_target(self) -> "TrainJobPayload":
        """Keep local jobs and pinned remote endpoints mutually consistent."""
        if self.training_target is TrainingTarget.LOCAL:
            if (
                self.remote_trainer_id is not None
                or self.remote_trainer_url is not None
                or self.remote_trainer_name is not None
            ):
                raise ValueError("Local training cannot specify a remote trainer")
            return self
        if self.remote_trainer_id is None:
            raise ValueError("Remote training requires remote_trainer_id")
        return self

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)

    @field_serializer("dataset_id")
    def serialize_dataset_id(self, dataset_id: UUID, _info: Any) -> str:
        return str(dataset_id)

    @field_serializer("base_model_id")
    def serialize_base_model_id(self, base_model_id: UUID | None, _info: Any) -> str | None:
        return str(base_model_id) if base_model_id else None

    @field_serializer("snapshot_id")
    def serialize_snapshot_id(self, snapshot_id: UUID | None, _info: Any) -> str | None:
        return str(snapshot_id) if snapshot_id else None

    @field_serializer("remote_job_id")
    def serialize_remote_job_id(self, remote_job_id: UUID | None, _info: Any) -> str | None:
        return str(remote_job_id) if remote_job_id else None


class TrainJob(BaseJob):
    type: Literal[JobType.TRAINING] = JobType.TRAINING  # type: ignore[valid-type]
    payload: TrainJobPayload


class DatasetImportJob(BaseJob):
    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT  # type: ignore[valid-type]
    payload: DatasetImportJobPayload


JobPayload = TrainJobPayload | DatasetImportJobPayload

Job = Annotated[
    TrainJob | DatasetImportJob,
    Field(discriminator="type"),
]

JobList.model_rebuild()

from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, model_validator

from schemas.base_job import BaseJob, JobType
from schemas.dataset_import_job import DatasetImportJobPayload
from schemas.hardware import DeviceType
from training.job import SNAPFLOW_POLICIES


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
    SSH = "ssh"


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
_DEFAULT_SNAPFLOW_DISTILL_EPOCHS = 3


class TrainJobPayloadBase(BaseModel):
    """Fields shared by every training execution target.

    Concrete payloads are `LocalTrainJobPayload`, `RemoteTrainJobPayload`, and
    `SshTrainJobPayload` below: each adds only the fields meaningful for its
    target and forbids the rest (`extra="forbid"`), so a payload can never
    express two targets at once and target-specific fields don't need a
    manual mutual-exclusion validator. Adding a target (e.g. a future
    AWS-provisioned trainer) means adding one subclass here and one entry in
    the `TrainJobPayload` union, not another branch in a validator.
    """

    model_config = ConfigDict(extra="forbid")

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
    def resolve_training_limit(self) -> "TrainJobPayloadBase":
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
    snapflow_enabled: bool = Field(
        default=False,
        description=(
            "Enable SnapFlow self-distillation, producing a policy that generates an action chunk "
            "in a single denoising step. This results in much faster inference but can reduce accuracy. "
            f"Only available for flow-matching policies (e.g. {sorted(SNAPFLOW_POLICIES)})."
        ),
    )
    snapflow_distill_epochs: int = Field(
        default=_DEFAULT_SNAPFLOW_DISTILL_EPOCHS,
        ge=1,
        le=10_000,
        description=(
            "How many additional epochs to spend distilling, appended after the "
            "max_epochs teacher run. Ignored when snapflow_enabled is false."
        ),
    )

    @model_validator(mode="after")
    def validate_snapflow(self) -> "TrainJobPayloadBase":
        """Reject a distillation request the policy cannot honour.

        Runs after ``resolve_training_limit``, so ``max_epochs`` is resolved.

        Returns:
            The validated payload.

        Raises:
            ValueError: If the policy has no SnapFlow implementation.
        """
        if not self.snapflow_enabled:
            return self

        if self.policy.lower() not in SNAPFLOW_POLICIES:
            msg = (
                f"SnapFlow distillation is not available for policy {self.policy!r}; "
                f"it requires a flow-matching policy ({sorted(SNAPFLOW_POLICIES)})."
            )
            raise ValueError(msg)
        return self

    @property
    def snapflow_start_epoch(self) -> int | None:
        """Epoch at which distillation begins, or None when it is disabled.

        Zero-based, matching ``SnapFlowPhaseCallback(start_epoch=...)``: with
        ``max_epochs=8`` and ``snapflow_distill_epochs=3``, epochs 0-7 train
        with flow matching and epochs 8-10 distill (11 epochs total).
        ``snapflow_distill_epochs`` is additive on top of ``max_epochs``, not
        carved out of it, so raising the teacher budget never shortens
        distillation (and vice versa).

        Returns:
            The phase boundary, or None for an ordinary flow-matching run.
        """
        if not self.snapflow_enabled:
            return None
        return self.max_epochs or _DEFAULT_MAX_EPOCHS

    @property
    def total_epochs(self) -> int:
        """Total training epochs, including the SnapFlow distillation phase.

        This is what should be handed to the trainer's epoch budget: the
        teacher phase always runs the full ``max_epochs``, and distillation
        (when enabled) extends the run rather than eating into it.

        Returns:
            ``max_epochs`` when SnapFlow is disabled, otherwise
            ``max_epochs + snapflow_distill_epochs``.
        """
        max_epochs = self.max_epochs or _DEFAULT_MAX_EPOCHS
        if not self.snapflow_enabled:
            return max_epochs
        return max_epochs + self.snapflow_distill_epochs

    # Set by the worker for every target (not just remote ones): a snapshot is
    # taken up front so a job's model has stable provenance, and a remote/SSH
    # run additionally needs its remote job id to reattach after a restart.
    remote_job_id: UUID | None = Field(
        default=None, description="Remote trainer job id, set when a remote run is in flight (for restart reattach)"
    )
    snapshot_id: UUID | None = Field(
        default=None, description="Dataset snapshot id retained while a remote run is in flight (for model provenance)"
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

    @field_serializer("snapshot_id")
    def serialize_snapshot_id(self, snapshot_id: UUID | None, _info: Any) -> str | None:
        return str(snapshot_id) if snapshot_id else None

    @field_serializer("remote_job_id")
    def serialize_remote_job_id(self, remote_job_id: UUID | None, _info: Any) -> str | None:
        return str(remote_job_id) if remote_job_id else None


class LocalTrainJobPayload(TrainJobPayloadBase):
    """Trains in the Studio process on a local device."""

    training_target: Literal[TrainingTarget.LOCAL] = TrainingTarget.LOCAL


class RemoteTrainJobPayload(TrainJobPayloadBase):
    """Offloads to a directly-configured remote trainer.

    `base_model_id` resume is rejected by `RemoteTrainingTargetHandler.prepare`
    (not here): the trainer protocol has no way to upload a base checkpoint,
    and that check raises a domain-specific `RemoteResumeUnsupportedError`
    rather than a generic schema validation error.
    """

    training_target: Literal[TrainingTarget.REMOTE] = TrainingTarget.REMOTE
    remote_trainer_id: UUID = Field(..., description="Configured remote trainer selected for a remote run")
    remote_trainer_url: str | None = Field(
        default=None,
        description="Resolved remote trainer URL pinned when the job is submitted for restart recovery",
    )
    remote_trainer_name: str | None = Field(
        default=None,
        description="Resolved remote trainer name pinned when the job is submitted, for display in job logs",
    )


class SshTrainJobPayload(TrainJobPayloadBase):
    """Trains on an SSH-provisioned remote server.

    `base_model_id` resume is rejected by `SshTrainingTargetHandler.prepare`
    for the same reason as `RemoteTrainJobPayload`.
    """

    training_target: Literal[TrainingTarget.SSH] = TrainingTarget.SSH
    remote_server_id: UUID = Field(..., description="Configured SSH-provisioned remote server selected for an SSH run")


TrainJobPayload = Annotated[
    LocalTrainJobPayload | RemoteTrainJobPayload | SshTrainJobPayload,
    Field(discriminator="training_target"),
]

# Used to (de)serialize a persisted/request payload dict into the right
# variant, since `TrainJobPayload` is a type alias (not a class) and has no
# `model_validate`/`model_dump` of its own.
TrainJobPayloadAdapter: TypeAdapter[LocalTrainJobPayload | RemoteTrainJobPayload | SshTrainJobPayload] = TypeAdapter(
    TrainJobPayload
)


class TrainJob(BaseJob):
    type: Literal[JobType.TRAINING] = JobType.TRAINING  # type: ignore[valid-type]
    payload: TrainJobPayload


class DatasetImportJob(BaseJob):
    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT  # type: ignore[valid-type]
    payload: DatasetImportJobPayload


JobPayload = LocalTrainJobPayload | RemoteTrainJobPayload | SshTrainJobPayload | DatasetImportJobPayload

Job = Annotated[
    TrainJob | DatasetImportJob,
    Field(discriminator="type"),
]

JobList.model_rebuild()

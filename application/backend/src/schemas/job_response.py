"""Public job snapshots, separate from restart-safe persisted worker payloads."""

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, TypeAdapter, model_validator

from schemas.base_job import BaseJob, JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload
from schemas.job import Job as InternalJob
from schemas.job import LocalTrainJobPayload, OptimizeModelJobPayload, TrainingTarget, TrainJobPayloadBase
from schemas.operational_issue import OperationalIssue


class RemoteTrainJobPayloadResponse(TrainJobPayloadBase):
    """Remote training details without provider-specific authentication state."""

    training_target: Literal[TrainingTarget.REMOTE] = TrainingTarget.REMOTE  # type: ignore[valid-type]
    remote_trainer_id: UUID
    remote_trainer_url: str | None = None
    remote_trainer_name: str | None = None
    cancel_requested: bool | None = None


class JobResponseBase(BaseJob):
    """Project worker state into a provider-independent public snapshot."""

    issues: list[OperationalIssue]

    @model_validator(mode="before")
    @classmethod
    def from_worker_job(cls, value: Any) -> Any:
        from schemas.job import DatasetImportJob as InternalDatasetImportJob
        from schemas.job import OptimizeModelJob as InternalOptimizeModelJob
        from schemas.job import TrainJob as InternalTrainJob
        from services.aws_auth_service import operational_authentication_issue

        if not isinstance(value, (InternalTrainJob, InternalDatasetImportJob, InternalOptimizeModelJob)):
            return value
        data = value.model_dump(mode="json")
        issue = getattr(value.payload, "authentication_issue", None)
        data["payload"].pop("authentication_issue", None)
        data["issues"] = (
            [operational_authentication_issue(issue)]
            if issue is not None and value.status in {JobStatus.PENDING, JobStatus.RUNNING}
            else []
        )
        return data


class TrainJob(JobResponseBase):
    """Training job API response."""

    type: Literal[JobType.TRAINING] = JobType.TRAINING  # type: ignore[valid-type]
    payload: Annotated[LocalTrainJobPayload | RemoteTrainJobPayloadResponse, Field(discriminator="training_target")]


class DatasetImportJob(JobResponseBase):
    """Dataset import job API response."""

    type: Literal[JobType.DATASET_IMPORT] = JobType.DATASET_IMPORT  # type: ignore[valid-type]
    payload: DatasetImportJobPayload


class OptimizeModelJob(JobResponseBase):
    """Model optimization job API response."""

    type: Literal[JobType.OPTIMIZE_MODEL] = JobType.OPTIMIZE_MODEL  # type: ignore[valid-type]
    payload: OptimizeModelJobPayload


JobResponse = Annotated[TrainJob | DatasetImportJob | OptimizeModelJob, Field(discriminator="type")]
JOB_RESPONSE_ADAPTER: TypeAdapter[TrainJob | DatasetImportJob | OptimizeModelJob] = TypeAdapter(JobResponse)


def job_response(job: InternalJob) -> TrainJob | DatasetImportJob | OptimizeModelJob:
    """Use the HTTP response contract for WebSocket snapshots as well."""
    return JOB_RESPONSE_ADAPTER.validate_python(job, from_attributes=True)

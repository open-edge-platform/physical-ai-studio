# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-job lifecycle policies; heavy worker imports stay inside child processes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sqlalchemy import func, or_

from db.schema import JobDB
from exceptions import InvalidJobStateError
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import ImportStep
from schemas.job import DatasetImportJob, TrainingTarget, TrainJobPayloadAdapter
from services.training_targets import target_key

if TYPE_CHECKING:
    import multiprocessing as mp
    from collections.abc import Callable, Mapping
    from multiprocessing.synchronize import Event as EventClass
    from uuid import UUID

    from sqlalchemy.sql.elements import ColumnElement

    from schemas import Job
    from services.job_service import JobService
    from workers.base import BaseProcessWorker


class JobHandler(ABC):
    """Scheduling, claim, cancellation and recovery policy for one job type.

    Workers retain business execution and result publication. Returning from a
    worker is not necessarily completion: imports can wait for review and remote
    training can requeue for reattachment.
    """

    job_type: JobType
    label: str
    supports_running_cancel = False

    def is_runnable(self, job: Job) -> bool:
        """Whether a queued job is ready to execute rather than waiting for input."""
        return job.status == JobStatus.PENDING

    @abstractmethod
    def resource_key(self, job: Job) -> str:
        """Return the exclusive resource reservation required by this job."""

    def resource_is_busy(self, resource: str, active: Mapping[str, object]) -> bool:
        """Whether another worker already holds this resource."""
        return resource in active

    @abstractmethod
    def create_worker(
        self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass | None, queue: mp.Queue
    ) -> BaseProcessWorker:
        """Construct the worker inside its child process."""

    def claim_filters(self) -> list[ColumnElement[bool]]:
        """Additional eligibility checks evaluated atomically with the claim."""
        return []

    def cancel_filters(self) -> list[ColumnElement[bool]]:
        """Conditions under which queued cancellation needs no worker cooperation."""
        return []

    async def claim(self, job_id: UUID, service: JobService) -> Job | None:
        """Atomically claim the assigned job if it is still runnable."""
        return await service.repo.transition_pending_job(
            job_id,
            JobStatus.RUNNING,
            self.job_type,
            message=f"Preparing {self.label.lower()}",
            expressions=self.claim_filters(),
        )

    async def cancel_pending(self, job_id: UUID, service: JobService) -> Job | None:
        """Cancel unclaimed work, preserving executions that need a worker to stop."""
        if not self.supports_running_cancel:
            raise InvalidJobStateError(f"{self.label} does not support generic interruption")
        return await service.repo.transition_pending_job(
            job_id,
            JobStatus.CANCELED,
            self.job_type,
            message=f"{self.label} canceled",
            expressions=self.cancel_filters(),
        )

    async def interrupt(self, job: Job, service: JobService, signal_cancel: Callable[[UUID], None]) -> None:
        """Coordinate queued cancellation with a concurrent worker claim."""
        if await service.cancel_pending_job(job.id, self.job_type) is None:
            signal_cancel(job.id)

    def needs_exit_recovery(self, assigned: Job, current: Job) -> bool:  # noqa: ARG002
        """Whether persisted state still belongs to the exited worker's execution."""
        return current.status == JobStatus.RUNNING or self.is_runnable(current)

    async def recover(self, job: Job, service: JobService) -> Job:
        """Reconcile an abandoned execution or a worker that failed before claiming."""
        reason = "exited" if job.status == JobStatus.PENDING else "stopped"
        return await service.update_job_status(
            job.id, JobStatus.FAILED, message=f"{self.label} worker {reason} before completing the job"
        )


class TrainingJobHandler(JobHandler):
    job_type = JobType.TRAINING
    label = "Training"
    supports_running_cancel = True

    def resource_key(self, job: Job) -> str:
        return target_key(TrainJobPayloadAdapter.validate_python(job.payload))

    def resource_is_busy(self, resource: str, active: Mapping[str, object]) -> bool:
        return super().resource_is_busy(resource, active) or (
            resource.startswith("remote:")
            and any(key.startswith(f"{resource}:") or resource.startswith(f"{key}:") for key in active)
        )

    def create_worker(
        self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass | None, queue: mp.Queue
    ) -> BaseProcessWorker:
        if cancel_event is None:
            raise ValueError("Training jobs require a cancellation event")
        from workers.training_worker import TrainingWorker

        return TrainingWorker(job_id, stop_event, cancel_event, queue)

    def cancel_filters(self) -> list[ColumnElement[bool]]:
        # A persisted remote execution must be canceled by a reattached worker.
        return [
            or_(
                func.json_extract(JobDB.payload, "$.training_target").is_distinct_from(TrainingTarget.REMOTE.value),
                func.json_extract(JobDB.payload, "$.remote_job_id").is_(None),
            )
        ]

    async def recover(self, job: Job, service: JobService) -> Job:
        from services.training_service import TrainingService

        return await TrainingService.recover_orphan_job(job, service)


_IMPORT_RUNNABLE_STEPS = (ImportStep.QUEUED_FOR_DETECTION, ImportStep.QUEUED_FOR_IMPORT)


class DatasetImportJobHandler(JobHandler):
    job_type = JobType.DATASET_IMPORT
    label = "Dataset import"

    def resource_key(self, job: Job) -> str:  # noqa: ARG002
        return "dataset_import"

    def is_runnable(self, job: Job) -> bool:
        return (
            super().is_runnable(job)
            and isinstance(job, DatasetImportJob)
            and job.payload.step in _IMPORT_RUNNABLE_STEPS
        )

    def claim_filters(self) -> list[ColumnElement[bool]]:
        return [func.json_extract(JobDB.payload, "$.step").in_([step.value for step in _IMPORT_RUNNABLE_STEPS])]

    def needs_exit_recovery(self, assigned: Job, current: Job) -> bool:
        if (
            current.status == JobStatus.PENDING
            and isinstance(assigned, DatasetImportJob)
            and isinstance(current, DatasetImportJob)
            and current.payload.step != assigned.payload.step
        ):
            # Review can be finalized before the detection process finishes exiting.
            return False
        return super().needs_exit_recovery(assigned, current)

    def create_worker(
        self,
        job_id: UUID,
        stop_event: EventClass,
        cancel_event: EventClass | None,  # noqa: ARG002
        queue: mp.Queue,
    ) -> BaseProcessWorker:
        from workers.dataset_import_worker import DatasetImportWorker

        return DatasetImportWorker(job_id, stop_event, queue)


class ModelOptimizationJobHandler(JobHandler):
    job_type = JobType.OPTIMIZE_MODEL
    label = "Model optimization"
    supports_running_cancel = True

    def resource_key(self, job: Job) -> str:  # noqa: ARG002
        return "local"

    def create_worker(
        self, job_id: UUID, stop_event: EventClass, cancel_event: EventClass | None, queue: mp.Queue
    ) -> BaseProcessWorker:
        if cancel_event is None:
            raise ValueError("Model optimization jobs require a cancellation event")
        from workers.model_optimization_worker import ModelOptimizationWorker

        return ModelOptimizationWorker(job_id, stop_event, cancel_event, queue)

    async def recover(self, job: Job, service: JobService) -> Job:
        from services.model_optimization_service import ModelOptimizationService

        ModelOptimizationService.cleanup(job.id)
        return await super().recover(job, service)


_HANDLERS: dict[JobType, JobHandler] = {
    handler.job_type: handler
    for handler in (TrainingJobHandler(), DatasetImportJobHandler(), ModelOptimizationJobHandler())
}


def get_job_handler(job_type: JobType) -> JobHandler:
    """Return the lifecycle policy registered for a job type."""
    try:
        return _HANDLERS[job_type]
    except KeyError:
        raise ValueError(f"Unsupported job type: {job_type}") from None

import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB
from exceptions import (
    DuplicateJobException,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    UnsupportedDeviceError,
)
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.job import JobPayload, TrainJob, TrainJobPayload
from services.system_service import SystemService


class JobService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = JobRepository(session)

    async def create_job(self, job: Job) -> Job:
        return await self.repo.save(job)

    async def get_job_list(self, extra_filters: dict | None = None) -> list[Job]:
        return await self.repo.get_all(extra_filters=extra_filters)

    async def get_job_by_id(self, job_id: UUID) -> Job:
        job = await self.repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
        return job

    async def get_jobs_by_ids(self, job_ids: list[UUID]) -> list[Job]:
        """Fetch multiple jobs by id in a single query."""
        if not job_ids:
            return []

        return await self.repo.get_all(expressions=[JobDB.id.in_([str(job_id) for job_id in job_ids])])

    async def submit_train_job(self, payload: TrainJobPayload) -> Job:
        # Validate the requested training device before persisting the job
        if payload.device is not None and not SystemService.is_device_supported_for_training(payload.device.type):
            raise UnsupportedDeviceError(
                device_type=payload.device.type,
                supported=SystemService.supported_training_device_types(),
            )

        if await self.repo.is_job_duplicate(project_id=payload.project_id, payload=payload):
            raise DuplicateJobException

        try:
            job = TrainJob(
                project_id=payload.project_id,
                payload=payload,
                message="Training job submitted",
            )
            return await self.repo.save(job)
        except IntegrityError:
            raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=payload.project_id)

    async def get_pending_train_job(self) -> Job | None:
        return await self.repo.get_pending_job_by_type(JobType.TRAINING)

    async def update_job_payload(
        self,
        job_id: UUID,
        payload: JobPayload,
        *,
        status: JobStatus | None = None,
        message: str | None = None,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> Job:
        job = await self.repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, resource_id=job_id)

        updates: dict = {"payload": payload.model_dump(mode="json")}
        if status is not None:
            updates["status"] = status
        if message is not None:
            updates["message"] = message
        if progress is not None:
            updates["progress"] = progress
        if extra_info is not None:
            updates["extra_info"] = extra_info
        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
            updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)

        return await self.repo.update(job, updates)

    async def update_job_status(
        self,
        job_id: UUID,
        status: JobStatus,
        message: str | None = None,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> Job:
        job = await self.repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, resource_id=job_id)
        updates: dict = {"status": status}
        if message is not None:
            updates["message"] = message
        progress_ = 100 if status is JobStatus.COMPLETED else progress
        if progress_ is not None:
            updates["progress"] = progress_
        if extra_info is not None:
            updates["extra_info"] = extra_info
        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
            updates["end_time"] = datetime.datetime.now(tz=datetime.UTC)
        return await self.repo.update(job, updates)

    async def update_job(self, job: Job, update: dict) -> Job:
        return await self.repo.update(job, update)

    async def delete_job(self, job_id: UUID) -> None:
        job: Job | None = await self.repo.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundError(ResourceType.JOB, str(job_id))

        if job.status not in {JobStatus.FAILED, JobStatus.CANCELED}:
            raise ResourceInUseError(ResourceType.JOB, str(job_id))

        await self.repo.delete_by_id(job_id)

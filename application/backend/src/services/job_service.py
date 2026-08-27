import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.schema import JobDB
from exceptions import DuplicateJobException, ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.job import JobPayload, TrainJob, TrainJobPayload
from services.remote_server_service import RemoteServerService
from services.remote_trainer_service import RemoteTrainerService
from services.training_targets import get_training_target_handler


class JobService:
    def __init__(
        self,
        session: AsyncSession,
        remote_trainer_service: RemoteTrainerService | None = None,
        remote_server_service: RemoteServerService | None = None,
    ) -> None:
        self.session = session
        self.repo = JobRepository(session)
        self.remote_trainer_service = remote_trainer_service
        self.remote_server_service = remote_server_service

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
        """Validate and persist a training job with its execution target pinned."""
        handler = get_training_target_handler(
            payload, self.session, self.remote_trainer_service, self.remote_server_service
        )
        payload = await handler.prepare(payload)

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

    async def get_pending_train_jobs(self) -> list[Job]:
        """Return pending training jobs in submission order."""
        return await self.repo.get_pending_jobs_by_type(JobType.TRAINING)

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

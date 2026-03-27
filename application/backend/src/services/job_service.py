import datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from db.schema import JobDB
from exceptions import DuplicateJobException, ResourceInUseError, ResourceNotFoundError, ResourceType
from repositories import JobRepository
from schemas import Job
from schemas.job import JobStatus, JobType, TrainJobPayload


class JobService:
    @staticmethod
    async def get_job_list(extra_filters: dict | None = None) -> list[Job]:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_all(extra_filters=extra_filters)

    @staticmethod
    async def get_job_by_id(job_id: UUID) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))
            return job

    @staticmethod
    async def get_jobs_by_ids(job_ids: list[UUID]) -> list[Job]:
        """Fetch multiple jobs by id in a single query."""
        if not job_ids:
            return []

        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_all(expressions=[JobDB.id.in_([str(job_id) for job_id in job_ids])])

    @staticmethod
    async def submit_train_job(payload: TrainJobPayload) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            if await repo.is_job_duplicate(project_id=payload.project_id, payload=payload):
                raise DuplicateJobException

            try:
                job = Job(
                    project_id=payload.project_id,
                    type=JobType.TRAINING,
                    payload=payload.model_dump(),
                    message="Training job submitted",
                )
                return await repo.save(job)
            except IntegrityError:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=payload.project_id)

    @staticmethod
    async def get_pending_train_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    @staticmethod
    async def update_job_status(
        job_id: UUID,
        status: JobStatus,
        message: str | None = None,
        progress: int | None = None,
        extra_info: dict | None = None,
    ) -> Job:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
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
            return await repo.update(job, updates)

    @staticmethod
    async def delete_job(job_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job: Job | None = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundError(ResourceType.JOB, str(job_id))

            if job.status != "failed":
                raise ResourceInUseError(ResourceType.JOB, str(job_id))

            await repo.delete_by_id(job_id)

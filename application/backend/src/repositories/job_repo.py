import datetime
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from db.schema import JobDB
from repositories.base import BaseRepository
from repositories.mappers import JobMapper
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.job import JobPayload


class JobRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, schema=JobDB)

    @property
    def to_schema(self) -> Callable[[Job], JobDB]:
        return JobMapper.to_schema

    @property
    def from_schema(self) -> Callable[[JobDB], Job]:
        return JobMapper.from_schema

    async def is_job_duplicate(self, project_id: UUID, payload: JobPayload) -> bool:
        # Match the JSON representation stored by JobMapper, including UUID values.
        payload_dict = payload.model_dump(mode="json")

        # Check for jobs with same payload that are not completed
        existing_job = await self.get_one(
            extra_filters={"project_id": self._id_to_str(project_id), "payload": payload_dict},
            expressions=[
                JobDB.status != JobStatus.COMPLETED,
                JobDB.status != JobStatus.FAILED,
                JobDB.status != JobStatus.CANCELED,
            ],
        )

        return existing_job is not None

    async def get_pending_job_by_type(self, job_type: JobType) -> Job | None:
        return await self.get_one(
            extra_filters={"type": job_type, "status": JobStatus.PENDING},
            order_by=self.schema.created_at,
            ascending=True,
        )

    async def get_pending_jobs(self, job_type: JobType | None = None) -> list[Job]:
        """Return queued jobs in submission order, including jobs waiting for input."""
        filters: dict[str, JobStatus | JobType] = {"status": JobStatus.PENDING}
        if job_type is not None:
            filters["type"] = job_type
        return await self.get_all(extra_filters=filters, order_by=self.schema.created_at, ascending=True)

    async def transition_pending_job(
        self,
        job_id: UUID,
        status: JobStatus,
        job_type: JobType,
        *,
        message: str,
        expressions: list[ColumnElement[bool]] | None = None,
    ) -> Job | None:
        """Atomically claim or cancel a queued job, never overwrite another state."""
        if status not in {JobStatus.RUNNING, JobStatus.CANCELED}:
            raise ValueError("Queued jobs can only be claimed or canceled")
        now = datetime.datetime.now(tz=datetime.UTC)
        values: dict = {"status": status, "message": message}
        if status == JobStatus.RUNNING:
            values["start_time"] = func.coalesce(JobDB.start_time, now)
        else:
            values["end_time"] = now
        query = update(JobDB).where(
            JobDB.id == str(job_id), JobDB.type == job_type, JobDB.status == JobStatus.PENDING, *(expressions or [])
        )
        result = await self.db.execute(query.values(**values).returning(JobDB.id))
        changed_id = result.scalar_one_or_none()
        await self.db.commit()
        return await self.get_by_id(changed_id) if changed_id is not None else None

    async def get_jobs_by_type(self, project_id: UUID, job_type: JobType) -> list[Job]:
        return await self.get_all(
            extra_filters={"project_id": self._id_to_str(project_id), "type": job_type},
        )

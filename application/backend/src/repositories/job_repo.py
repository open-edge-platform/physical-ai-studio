import datetime
from collections.abc import Callable
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import JobDB
from repositories.base import BaseRepository
from repositories.mappers import JobMapper
from schemas import Job
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import ImportStep
from schemas.job import TrainJobPayload


class JobRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, schema=JobDB)

    @property
    def to_schema(self) -> Callable[[Job], JobDB]:
        return JobMapper.to_schema

    @property
    def from_schema(self) -> Callable[[JobDB], Job]:
        return JobMapper.from_schema

    async def is_job_duplicate(self, project_id: UUID, payload: TrainJobPayload) -> bool:
        # Convert payload to dict for comparison
        payload_dict = payload.model_dump()

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

    async def claim_pending_dataset_import_job(self) -> Job | None:
        pending_steps = [
            ImportStep.QUEUED_FOR_DETECTION.value,
            ImportStep.QUEUED_FOR_IMPORT.value,
        ]

        query = (
            select(JobDB.id)
            .where(
                JobDB.type == JobType.DATASET_IMPORT,
                JobDB.status == JobStatus.PENDING,
                func.json_extract(JobDB.payload, "$.step").in_(pending_steps),
            )
            .order_by(JobDB.created_at.asc())
            .limit(1)
        )
        result = await self.db.execute(query)
        job_id = result.scalar_one_or_none()
        if job_id is None:
            return None

        claim = (
            update(JobDB)
            .where(
                JobDB.id == job_id,
                JobDB.status == JobStatus.PENDING,
                func.json_extract(JobDB.payload, "$.step").in_(pending_steps),
            )
            .values(
                status=JobStatus.RUNNING,
                start_time=datetime.datetime.now(tz=datetime.UTC),
            )
        )
        await self.db.execute(claim)
        await self.db.commit()

        return await self.get_by_id(job_id)

    async def get_jobs_by_type(self, project_id: UUID, job_type: JobType) -> list[Job]:
        return await self.get_all(
            extra_filters={"project_id": self._id_to_str(project_id), "type": job_type},
        )

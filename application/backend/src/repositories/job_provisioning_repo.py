from collections.abc import Callable
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import JobDB, JobProvisioningDB
from repositories.base import BaseRepository
from repositories.mappers.job_provisioning_mapper import JobProvisioningMapper
from schemas.base_job import JobStatus
from schemas.job_provisioning import JobProvisioning, JobProvisioningUpdate

# Jobs that can still own a live container. Anything else has finished, so its
# provisioning row only describes something already torn down.
_NON_TERMINAL_JOB_STATUSES = (JobStatus.PENDING.value, JobStatus.RUNNING.value)


class JobProvisioningRepository(BaseRepository[JobProvisioning, JobProvisioningDB]):
    """Persistence for per-job SSH provisioning state.

    The table is keyed by ``job_id``, not ``id``, so the ``id``-based helpers on
    ``BaseRepository`` (``get_by_id``, ``delete_by_id``, ``update``) do not apply.
    Use the ``*_by_job_id`` methods below instead.
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db, JobProvisioningDB)

    @property
    def to_schema(self) -> Callable[[JobProvisioning], JobProvisioningDB]:
        return JobProvisioningMapper.to_schema

    @property
    def from_schema(self) -> Callable[[JobProvisioningDB], JobProvisioning]:
        return JobProvisioningMapper.from_schema

    async def get_by_job_id(self, job_id: str | UUID) -> JobProvisioning | None:
        """Return the provisioning state recorded for one job."""
        return await self.get_one(extra_filters={"job_id": self._id_to_str(job_id)})

    async def update_by_job_id(self, job_id: str | UUID, update: JobProvisioningUpdate) -> JobProvisioning:
        """Merge the fields set on ``update`` into a job's provisioning row.

        Overrides the ``id``-keyed base ``update``: provisioning rows have no
        ``id`` column, so the base implementation's refresh-by-id cannot work.
        Unset fields are left alone rather than nulled, because each provisioning
        stage writes only what it just learned.
        """
        job_id = self._id_to_str(job_id)
        current = await self.get_by_job_id(job_id)
        if current is None:
            raise ValueError(f"No provisioning state recorded for job `{job_id}`")

        changes = update.model_dump(exclude_unset=True, exclude_none=True)
        merged = JobProvisioning.model_validate(dict(current.model_copy(update=changes, deep=True)))
        await self.db.merge(self.to_schema(merged))
        await self.db.commit()

        refreshed = await self.get_by_job_id(job_id)
        if refreshed is None:
            raise ValueError(f"Provisioning state for job `{job_id}` disappeared during update")
        return refreshed

    async def delete_by_job_id(self, job_id: str | UUID) -> None:
        """Remove a job's provisioning row once its container is gone."""
        query = delete(JobProvisioningDB).where(JobProvisioningDB.job_id == self._id_to_str(job_id))
        await self.db.execute(query)
        await self.db.commit()

    async def list_active(self) -> list[JobProvisioning]:
        """Return provisioning rows for jobs that can still own a container.

        Drives startup reattach and the orphan sweep: a row whose job has already
        reached a terminal state describes nothing left to reclaim.
        """
        query = (
            select(JobProvisioningDB)
            .join(JobDB, JobDB.id == JobProvisioningDB.job_id)
            .where(JobDB.status.in_(_NON_TERMINAL_JOB_STATUSES))
            .order_by(JobProvisioningDB.created_at.asc())
        )
        results = await self.db.execute(query)
        return [self.from_schema(model) for model in results.scalars().all()]

    async def list_for_server(self, remote_server_id: str | UUID) -> list[JobProvisioning]:
        """Return every provisioning row that targets one server."""
        query = (
            select(JobProvisioningDB)
            .where(JobProvisioningDB.remote_server_id == self._id_to_str(remote_server_id))
            .order_by(JobProvisioningDB.created_at.asc())
        )
        results = await self.db.execute(query)
        return [self.from_schema(model) for model in results.scalars().all()]

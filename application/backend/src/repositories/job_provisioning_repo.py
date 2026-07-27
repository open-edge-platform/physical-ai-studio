from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql import expression

from db.schema import JobProvisioningDB
from repositories.base import BaseRepository
from repositories.mappers.job_provisioning_mapper import JobProvisioningMapper
from schemas.job_provisioning import JobProvisioning


class JobProvisioningRepository(BaseRepository[JobProvisioning, JobProvisioningDB]):
    """Persistence for per-job SSH provisioning state.

    The table's primary key is `job_id`, not `id`, so this repository does
    not use :meth:`BaseRepository.get_by_id`/`update`/`delete_by_id`
    (which assume an `id` column); it provides `job_id`-keyed equivalents.
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
        """Return the provisioning state for a job, if any has been recorded."""
        return await self.get_one(extra_filters={"job_id": self._id_to_str(job_id)})

    async def upsert(self, item: JobProvisioning) -> JobProvisioning:
        """Insert or update provisioning state, keyed by `job_id`."""
        schema_item = self.to_schema(item)
        await self.db.merge(schema_item)
        await self.db.commit()

        updated = await self.get_by_job_id(item.job_id)
        if updated is None:
            raise ValueError(f"JobProvisioning with job_id `{item.job_id}` doesn't exist after upsert")
        return updated

    async def delete_by_job_id(self, job_id: str | UUID) -> None:
        """Remove provisioning state for a job."""
        job_id_str = self._id_to_str(job_id)
        query = expression.delete(self.schema).where(self.schema.job_id == job_id_str)
        await self.db.execute(query)
        await self.db.commit()

from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Job


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(job: Job) -> JobDB:
        return JobDB(**job.model_dump())

    @staticmethod
    def from_schema(job_db: JobDB) -> Job:
        return Job.model_validate(job_db, from_attributes=True)

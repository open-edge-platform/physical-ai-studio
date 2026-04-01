from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Job


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Job) -> JobDB:
        return JobDB(**db_schema.model_dump())

    @staticmethod
    def from_schema(model: JobDB) -> Job:
        return Job.model_validate(model, from_attributes=True)

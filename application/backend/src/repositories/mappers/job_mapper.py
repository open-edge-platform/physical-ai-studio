from pydantic import TypeAdapter

from db.schema import JobDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Job

JOB_ADAPTER = TypeAdapter(Job)


class JobMapper(IBaseMapper):
    @staticmethod
    def to_schema(db_schema: Job) -> JobDB:
        return JobDB(**db_schema.model_dump())

    @staticmethod
    def from_schema(model: JobDB) -> Job:
        return JOB_ADAPTER.validate_python(model, from_attributes=True)

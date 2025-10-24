from db.schema import ProjectConfigDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import ProjectConfig


class ProjectConfigMapper(IBaseMapper):
    @staticmethod
    def to_schema(project_config: ProjectConfig) -> ProjectConfigDB:
        return ProjectConfigDB(**project_config.model_dump(mode="json"))

    @staticmethod
    def from_schema(model_db: ProjectConfigDB) -> ProjectConfig:
        return ProjectConfig.model_validate(model_db, from_attributes=True)
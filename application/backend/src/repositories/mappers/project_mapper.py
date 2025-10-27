from db.schema import ProjectDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Project
from .project_config_mapper import ProjectConfigMapper
from .dataset_mapper import DatasetMapper


class ProjectMapper(IBaseMapper):
    @staticmethod
    def to_schema(project: Project) -> ProjectDB:
        return ProjectDB(
            id=str(project.id),
            name=project.name,
            config=ProjectConfigMapper.to_schema(project.config),
            datasets=[DatasetMapper.to_schema(dataset) for dataset in project.datasets],
        )

    @staticmethod
    def from_schema(model_db: ProjectDB) -> Project:
        return Project.model_validate(model_db, from_attributes=True)

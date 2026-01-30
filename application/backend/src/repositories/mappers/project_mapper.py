from db.schema import ProjectDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Project

from .dataset_mapper import DatasetMapper


class ProjectMapper(IBaseMapper):
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project: Project) -> ProjectDB:
        """Convert Project schema to db model."""
        return ProjectDB(
            id=str(project.id),
            name=project.name,
            datasets=[DatasetMapper.to_schema(dataset) for dataset in project.datasets],
        )

    @staticmethod
    def from_schema(project_db: ProjectDB) -> Project:
        """Convert Project db entity to schema."""
        return Project.model_validate(
            {
                "id": project_db.id,
                "name": project_db.name,
                "updated_at": project_db.updated_at,
                "datasets": [DatasetMapper.from_schema(dataset) for dataset in project_db.datasets],
            }
        )

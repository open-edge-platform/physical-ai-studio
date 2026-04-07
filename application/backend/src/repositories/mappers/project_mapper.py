from db.schema import ProjectDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import Project

from .dataset_mapper import DatasetMapper


class ProjectMapper(IBaseMapper):
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Project) -> ProjectDB:
        """Convert Project schema to db model."""
        return ProjectDB(
            id=str(db_schema.id),
            name=db_schema.name,
            datasets=[DatasetMapper.to_schema(dataset) for dataset in db_schema.datasets],
        )

    @staticmethod
    def from_schema(model: ProjectDB) -> Project:
        """Convert Project db entity to schema."""
        return Project.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "updated_at": model.updated_at,
                "datasets": [DatasetMapper.from_schema(dataset) for dataset in model.datasets],
            }
        )

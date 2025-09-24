from db.schema import ProjectDB
from .config_mapper import ProjectConfigMapper
from .datasets_mapper import DatasetMapper
from schemas import Project


class ProjectMapper:
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_db: ProjectDB) -> Project:
        """Convert Project db entity to schema."""
        return Project.model_validate({
            "id": project_db.id,
            "name": project_db.name,
            "updated_at": project_db.updated_at,
            "config": ProjectConfigMapper.to_schema(project_db.config),
            "datasets": [DatasetMapper.to_schema(dataset) for dataset in project_db.datasets],
        })

    @staticmethod
    def from_schema(project: Project) -> Project:
        """Convert Project schema to db model."""
        return ProjectDB(
            id=str(project.id),
            name=project.name,
            config=ProjectConfigMapper.from_schema(project.config),
            datasets=[DatasetMapper.from_schema(dataset) for dataset in project.datasets],

        )

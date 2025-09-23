from db.schema import ProjectDB
from schemas import Project


class ProjectMapper:
    """Mapper for Project schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_db: ProjectDB) -> Project:
        """Convert Project db entity to schema."""
        return Project.model_validate({"id": project_db.id, "name": project_db.name, "updated_at": project_db.updated_at})

    @staticmethod
    def from_schema(project: Project) -> ProjectDB:
        """Convert Project schema to db model."""

        return ProjectDB(
            id=str(project.id),
            name=project.name,
        )

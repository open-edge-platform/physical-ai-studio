from db.schema import ProjectConfigDB
from schemas import ProjectConfig


class ProjectConfigMapper:
    """Mapper for Label schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_config_db: ProjectConfigDB | None) -> ProjectConfig | None:
        """Convert Label db entity to schema."""
        if project_config_db is None:
            return

        return ProjectConfig.model_validate(project_config_db, from_attributes=True)

    @staticmethod
    def from_schema(config: ProjectConfig) -> ProjectConfigDB:
        """Convert Label schema to db model."""

        return ProjectConfigDB(**config.model_dump(mode="json"))

from db.schema import ProjectConfigDB
from schemas import ProjectConfig, LeRobotDatasetInfo


class ProjectConfigMapper:
    """Mapper for Label schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_config_db: ProjectConfigDB | None) -> ProjectConfig | None:
        """Convert Label db entity to schema."""
        if project_config_db is None:
            return

        return ProjectConfig.model_validate(project_config_db, from_attributes=True)

    @staticmethod
    def from_schema(config: ProjectConfig | None) -> ProjectConfigDB | None:
        """Convert Label schema to db model."""
        if config is None:
            return
        return ProjectConfigDB(**config.model_dump(mode="json"))

    @staticmethod
    def from_lerobot_dataset(dataset: LeRobotDatasetInfo) -> ProjectConfig:
        """Create a config from a lerobot dataset."""
        return ProjectConfig(
            fps=dataset.fps
        )

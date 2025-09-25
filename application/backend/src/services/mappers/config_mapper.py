from db.schema import ProjectConfigDB
from schemas import ProjectConfig

from .camera_config_mapper import CameraConfigMapper


class ProjectConfigMapper:
    """Mapper for ProjectConfig schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_config_db: ProjectConfigDB | None) -> ProjectConfig | None:
        """Convert ProjectConfig db entity to schema."""
        if project_config_db is None:
            return None

        return ProjectConfig.model_validate(
            {
                "fps": project_config_db.fps,
                "cameras": [CameraConfigMapper.to_schema(camera) for camera in project_config_db.cameras],
            },
            from_attributes=True,
        )

    @staticmethod
    def from_schema(config: ProjectConfig | None) -> ProjectConfigDB | None:
        """Convert ProjectConfig schema to db model."""
        if config is None:
            return None
        return ProjectConfigDB(
            fps=config.fps, cameras=[CameraConfigMapper.from_schema(camera) for camera in config.cameras]
        )

from db.schema import ProjectConfigDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from repositories.mappers.camera_config_mapper import CameraConfigMapper
from schemas import ProjectConfig


class ProjectConfigMapper(IBaseMapper):
    """Mapper for ProjectConfig schema entity <-> DB entity conversions."""

    @staticmethod
    def from_schema(project_config_db: ProjectConfigDB | None) -> ProjectConfig | None:
        """Convert ProjectConfig db entity to schema."""
        if project_config_db is None:
            return None

        return ProjectConfig.model_validate(
            {
                "id": project_config_db.id,
                "fps": project_config_db.fps,
                "robot_type": project_config_db.robot_type,
                "cameras": [CameraConfigMapper.from_schema(camera) for camera in project_config_db.cameras],
            },
            from_attributes=True,
        )

    @staticmethod
    def to_schema(config: ProjectConfig | None) -> ProjectConfigDB | None:
        """Convert ProjectConfig schema to db model."""
        if config is None:
            return None

        return ProjectConfigDB(
            fps=config.fps,
            cameras=[CameraConfigMapper.to_schema(camera) for camera in config.cameras],
            robot_type=config.robot_type,
        )

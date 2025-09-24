from db.schema import ProjectConfigDB
from schemas import ProjectConfig, LeRobotDatasetInfo
from .camera_config_mapper import CameraConfigMapper


class ProjectConfigMapper:
    """Mapper for ProjectConfig schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(project_config_db: ProjectConfigDB | None) -> ProjectConfig | None:
        """Convert ProjectConfig db entity to schema."""
        if project_config_db is None:
            return

        return ProjectConfig.model_validate(
            {
                "fps": project_config_db.fps,
                "cameras": [
                    CameraConfigMapper.to_schema(camera)
                    for camera in project_config_db.cameras
                ],
            },
            from_attributes=True,
        )

    @staticmethod
    def from_schema(config: ProjectConfig | None) -> ProjectConfigDB | None:
        """Convert ProjectConfig schema to db model."""
        if config is None:
            return
        return ProjectConfigDB(
            fps=config.fps,
            cameras=[CameraConfigMapper.from_schema(camera) for camera in config.cameras]
        )

    @staticmethod
    def from_lerobot_dataset(dataset: LeRobotDatasetInfo) -> ProjectConfig:
        """Create a config from a lerobot dataset."""
        return ProjectConfig(
            fps=dataset.fps,
        )

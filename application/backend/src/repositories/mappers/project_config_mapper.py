from db.schema import ProjectConfigDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import ProjectConfig
from .camera_config_mapper import CameraConfigMapper


class ProjectConfigMapper(IBaseMapper):
    @staticmethod
    def to_schema(project_config: ProjectConfig) -> ProjectConfigDB:
        if project_config is None:
            return None

        return ProjectConfigDB(
            fps=project_config.fps,
            cameras=[CameraConfigMapper.to_schema(camera) for camera in project_config.cameras],
            robot_type=project_config.robot_type,
        )

    @staticmethod
    def from_schema(model_db: ProjectConfigDB) -> ProjectConfig:
        return ProjectConfig.model_validate(model_db, from_attributes=True)

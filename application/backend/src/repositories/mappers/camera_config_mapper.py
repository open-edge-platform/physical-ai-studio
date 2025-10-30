from db.schema import CameraConfigDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas import CameraConfig


class CameraConfigMapper(IBaseMapper):
    """Mapper for Camera schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(camera_config: CameraConfig) -> CameraConfigDB:
        return CameraConfigDB(**camera_config.model_dump(mode="json"))

    @staticmethod
    def from_schema(model_db: CameraConfigDB) -> CameraConfig:
        return CameraConfig.model_validate(model_db, from_attributes=True)

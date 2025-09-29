from db.schema import CameraConfigDB
from schemas import CameraConfig


class CameraConfigMapper:
    """Mapper for Camera schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(camera_config_db: CameraConfigDB | None) -> CameraConfig | None:
        """Convert db entity to schema."""
        if camera_config_db is None:
            return None

        return CameraConfig.model_validate(camera_config_db, from_attributes=True)

    @staticmethod
    def from_schema(config: CameraConfig | None) -> CameraConfigDB | None:
        """Convert schema to db model."""
        if config is None:
            return None
        return CameraConfigDB(**config.model_dump(mode="json"))

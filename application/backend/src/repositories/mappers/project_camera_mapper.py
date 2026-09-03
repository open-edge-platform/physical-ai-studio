import json

from db.schema import ProjectCameraDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.project_camera import Camera, CameraAdapter


class ProjectCameraMapper(IBaseMapper):
    """Mapper for Camera schema entity <-> DB entity conversions."""

    @staticmethod
    def to_schema(db_schema: Camera) -> ProjectCameraDB:
        """Convert Camera schema to db model."""
        if not db_schema.fingerprint:
            raise ValueError("Camera must be reselected")
        return ProjectCameraDB(
            id=str(db_schema.id),
            project_id="",  # Will be set by repository
            name=db_schema.name,
            driver=db_schema.driver,
            fingerprint=json.dumps(db_schema.fingerprint, sort_keys=True, separators=(",", ":")),
            hardware_name=db_schema.hardware_name,
            payload=db_schema.payload.model_dump(exclude_none=True),
            created_at=db_schema.created_at,
            updated_at=db_schema.updated_at,
        )

    @staticmethod
    def from_schema(model: ProjectCameraDB) -> Camera:
        """Convert Camera db entity to schema."""
        try:
            fingerprint = json.loads(model.fingerprint)
        except (TypeError, ValueError):
            fingerprint = None
        if not isinstance(fingerprint, dict) or not fingerprint:
            fingerprint = None

        return CameraAdapter.validate_python(
            {
                "id": model.id,
                "driver": model.driver,
                "name": model.name,
                "fingerprint": fingerprint,
                "hardware_name": model.hardware_name,
                "payload": model.payload,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )

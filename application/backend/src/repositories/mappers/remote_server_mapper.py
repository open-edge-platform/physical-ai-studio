from db.schema import RemoteServerDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.remote_server import RemoteServer


class RemoteServerMapper(IBaseMapper):
    """Map persisted SSH-provisioned training servers to API schemas."""

    @staticmethod
    def to_schema(db_schema: RemoteServer) -> RemoteServerDB:
        """Convert an API schema to its database model."""
        return RemoteServerDB(
            id=str(db_schema.id),
            name=db_schema.name,
            ssh_host_alias=db_schema.ssh_host_alias,
            device_type=db_schema.device_type.value,
            last_check_status=db_schema.last_check_status,
            last_check_at=db_schema.last_check_at,
            last_check_latency_ms=db_schema.last_check_latency_ms,
            last_check_reason_code=db_schema.last_check_reason_code,
        )

    @staticmethod
    def from_schema(model: RemoteServerDB) -> RemoteServer:
        """Convert a database model to its API schema."""
        return RemoteServer.model_validate(
            {
                "id": model.id,
                "name": model.name,
                "ssh_host_alias": model.ssh_host_alias,
                "device_type": model.device_type,
                "last_check_status": model.last_check_status,
                "last_check_at": model.last_check_at,
                "last_check_latency_ms": model.last_check_latency_ms,
                "last_check_reason_code": model.last_check_reason_code,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )

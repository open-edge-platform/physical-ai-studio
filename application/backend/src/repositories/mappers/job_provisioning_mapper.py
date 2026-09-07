from db.schema import JobProvisioningDB
from repositories.mappers.base_mapper_interface import IBaseMapper
from schemas.job_provisioning import JobProvisioning


class JobProvisioningMapper(IBaseMapper):
    """Map persisted per-job provisioning state to API schemas."""

    @staticmethod
    def to_schema(db_schema: JobProvisioning) -> JobProvisioningDB:
        """Convert an API schema to its database model."""
        return JobProvisioningDB(
            job_id=str(db_schema.job_id),
            remote_server_id=str(db_schema.remote_server_id),
            ssh_host_alias=db_schema.ssh_host_alias,
            image_ref=db_schema.image_ref,
            image_fallback_reason=db_schema.image_fallback_reason,
            image_digest=db_schema.image_digest,
            container_id=db_schema.container_id,
            container_name=db_schema.container_name,
            remote_port=db_schema.remote_port,
            local_tunnel_port=db_schema.local_tunnel_port,
            backend_instance_id=db_schema.backend_instance_id,
            trainer_build_version=db_schema.trainer_build_version,
            trainer_protocol_version=db_schema.trainer_protocol_version,
        )

    @staticmethod
    def from_schema(model: JobProvisioningDB) -> JobProvisioning:
        """Convert a database model to its API schema."""
        return JobProvisioning.model_validate(
            {
                "job_id": model.job_id,
                "remote_server_id": model.remote_server_id,
                "ssh_host_alias": model.ssh_host_alias,
                "image_ref": model.image_ref,
                "image_fallback_reason": model.image_fallback_reason,
                "image_digest": model.image_digest,
                "container_id": model.container_id,
                "container_name": model.container_name,
                "remote_port": model.remote_port,
                "local_tunnel_port": model.local_tunnel_port,
                "backend_instance_id": model.backend_instance_id,
                "trainer_build_version": model.trainer_build_version,
                "trainer_protocol_version": model.trainer_protocol_version,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
            }
        )

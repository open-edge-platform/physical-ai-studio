"""Per-job SSH provisioning state for a dynamically launched trainer container.

Persisted so a crashed backend can reclaim (or an operator can diagnose) an
orphaned container by its recorded id/name and remote/tunnel ports. See
``remote-ssh-trainer-plan.md`` step 4 (SSH container provisioning service).
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer


class JobProvisioning(BaseModel):
    """Provisioning state for one training job's SSH-launched trainer container."""

    job_id: UUID
    remote_server_id: UUID

    image_ref: str | None = Field(default=None, description="Resolved image reference (Git-SHA tag or 'latest')")
    image_fallback_reason: str | None = Field(
        default=None, description="Why 'latest' was used instead of a Git-SHA tag, null when the SHA tag resolved"
    )
    image_digest: str | None = Field(default=None, description="Immutable resolved image digest the container ran by")

    container_id: str | None = None
    container_name: str | None = None
    remote_port: int | None = Field(default=None, ge=1, le=65535, description="Loopback-only port on the remote host")
    local_tunnel_port: int | None = Field(default=None, ge=1, le=65535, description="Local end of the SSH tunnel")

    trainer_build_version: str | None = None
    trainer_protocol_version: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_serializer("job_id")
    def serialize_job_id(self, job_id: UUID, _info: Any) -> str:
        return str(job_id)

    @field_serializer("remote_server_id")
    def serialize_remote_server_id(self, remote_server_id: UUID, _info: Any) -> str:
        return str(remote_server_id)

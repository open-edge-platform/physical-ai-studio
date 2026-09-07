# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Per-job SSH provisioning state.

Stored in its own table keyed by ``job_id`` rather than inside the job payload
JSON, so a crashed backend can sweep or reclaim an orphaned container from
durable, queryable columns instead of parsing every job's payload.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from schemas.remote_server import SSH_HOST_ALIAS_PATTERN


class JobProvisioning(BaseModel):
    """What Studio provisioned on a remote server for one job.

    Written before the job accepts work so startup reattach and the orphan sweep
    can always find the container. Holds no credential: ``ssh_host_alias`` is the
    name of a Host stanza in the user's SSH config.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: UUID
    remote_server_id: UUID
    ssh_host_alias: str = Field(min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)

    # Which image was selected, why, and the immutable digest it resolved to.
    # The container always launches by digest, never by the mutable tag.
    image_ref: str | None = Field(default=None, max_length=512)
    image_fallback_reason: str | None = Field(
        default=None,
        max_length=512,
        description="Set when the protocol-<N> tag could not be resolved and `latest` was used instead.",
    )
    image_digest: str | None = Field(default=None, max_length=255)

    container_id: str | None = Field(default=None, max_length=128)
    container_name: str | None = Field(default=None, max_length=255)

    # Trainer port published on the remote host's loopback, and the local end of
    # the SSH forward. The local port is re-assigned on every reattach.
    remote_port: int | None = Field(default=None, ge=1, le=65535)
    local_tunnel_port: int | None = Field(default=None, ge=1, le=65535)

    # Per-process ownership marker. Remote servers are global and two Studio
    # instances can legitimately target the same host, so the orphan sweep must
    # prove ownership rather than assume it from the management labels alone.
    backend_instance_id: str | None = Field(default=None, max_length=255)

    # Reported by the provisioned trainer's /health, recorded for diagnosis.
    trainer_build_version: str | None = Field(default=None, max_length=255)
    trainer_protocol_version: int | None = Field(default=None, ge=0)

    created_at: datetime | None = None
    updated_at: datetime | None = None


class JobProvisioningUpdate(BaseModel):
    """Mutable provisioning state, written as each stage completes."""

    image_ref: str | None = Field(default=None, max_length=512)
    image_fallback_reason: str | None = Field(default=None, max_length=512)
    image_digest: str | None = Field(default=None, max_length=255)
    container_id: str | None = Field(default=None, max_length=128)
    container_name: str | None = Field(default=None, max_length=255)
    remote_port: int | None = Field(default=None, ge=1, le=65535)
    local_tunnel_port: int | None = Field(default=None, ge=1, le=65535)
    backend_instance_id: str | None = Field(default=None, max_length=255)
    trainer_build_version: str | None = Field(default=None, max_length=255)
    trainer_protocol_version: int | None = Field(default=None, ge=0)

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for SSH-provisioned remote training servers.

Studio stores no SSH credentials. A remote server is identified by the name of a
``Host`` stanza in the user's own ``~/.ssh/config``; ``asyncssh`` resolves that
alias and authenticates. No field here holds a key, password, or passphrase, and
none ever will - ``tests/schemas/test_remote_server.py`` asserts that.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.hardware import DeviceType

# Devices a remote trainer image exists for.
SSH_SERVER_DEVICE_TYPES = frozenset({DeviceType.CUDA, DeviceType.XPU})

# Health of the last recorded preflight. ``unknown`` means never checked.
RemoteServerCheckStatus = Literal["healthy", "degraded", "unreachable", "unknown"]

# An SSH config alias. Deliberately narrow: the value is interpolated into no
# shell string anywhere, but a strict charset keeps it out of argument arrays
# that a future change might pass to a shell, and rejects wildcard patterns.
SSH_HOST_ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$"


class RemoteServerCreate(BaseModel):
    """User-supplied configuration for an SSH-provisioned training server."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    ssh_host_alias: str = Field(
        min_length=1,
        max_length=255,
        pattern=SSH_HOST_ALIAS_PATTERN,
        description="Name of a Host stanza in the user's SSH config. Non-secret.",
    )
    device_type: DeviceType = Field(description="Accelerator on the server. Only cuda and xpu are supported.")

    @field_validator("device_type")
    @classmethod
    def _reject_unsupported_device(cls, value: DeviceType) -> DeviceType:
        """Reject devices with no published trainer image."""
        if value not in SSH_SERVER_DEVICE_TYPES:
            supported = ", ".join(sorted(device.value for device in SSH_SERVER_DEVICE_TYPES))
            raise ValueError(f"device_type must be one of: {supported}")
        return value


class RemoteServerUpdate(BaseModel):
    """Mutable fields for an SSH-provisioned training server."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=255)
    ssh_host_alias: str | None = Field(default=None, min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)
    device_type: DeviceType | None = None

    @field_validator("device_type")
    @classmethod
    def _reject_unsupported_device(cls, value: DeviceType | None) -> DeviceType | None:
        """Reject devices with no published trainer image."""
        if value is not None and value not in SSH_SERVER_DEVICE_TYPES:
            supported = ", ".join(sorted(device.value for device in SSH_SERVER_DEVICE_TYPES))
            raise ValueError(f"device_type must be one of: {supported}")
        return value


class RemoteServer(RemoteServerCreate):
    """A persisted SSH-provisioned training server.

    ``last_check_*`` carries the summary of the most recent preflight so a
    transient failure marks the record unhealthy instead of destroying it.
    """

    id: UUID
    last_check_status: RemoteServerCheckStatus = "unknown"
    last_check_at: datetime | None = None
    last_check_latency_ms: int | None = Field(default=None, ge=0)
    last_check_reason_code: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ResolvedSshHost(BaseModel):
    """The effective connection target an alias resolves to, for display only.

    Derived from the SSH config at read time rather than persisted, so a stored
    server can never disagree with the config it is defined by. Carries no
    ``IdentityFile``, ``IdentityAgent``, ``CertificateFile``, or password: the
    reader must not surface credential material even in resolved form.
    """

    alias: str
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = None
    found: bool = Field(description="False when the alias is absent from the SSH config or matches only a wildcard.")


class RemoteServerWithResolution(RemoteServer):
    """A persisted server plus the connection target its alias resolves to now."""

    resolved: ResolvedSshHost


class SshHostAliasOption(BaseModel):
    """A selectable SSH host alias for the create/edit form."""

    alias: str
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = None

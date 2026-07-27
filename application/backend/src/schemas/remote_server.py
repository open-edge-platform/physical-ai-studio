"""Schemas for registered SSH-accessible remote training servers.

`ssh_secret`, `ssh_key_passphrase`, and `host_key` are confidential/internal and must never
be returned from a service's public API or an HTTP response; only
:class:`RemoteServerInternal` (used by the repository/mapper and the future
SSH provisioning boundary) carries them.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from .hardware import DeviceType

# Remote servers are provisioned for GPU/XPU training only; CPU/NPU are not
# supported connection targets.
RemoteServerDeviceType = Literal[DeviceType.CUDA, DeviceType.XPU]


class SSHAuthType(StrEnum):
    """Supported SSH authentication mechanisms for a remote server."""

    KEY = "key"
    PASSWORD = "password"  # noqa: S105 # nosec B105 -- enum member, not a credential


class LastCheckSummary(BaseModel):
    """Cached result of the most recent SSH preflight, if one has ever run."""

    status: Literal["healthy", "degraded", "unreachable"] | None = Field(
        default=None, description="Result of the most recent preflight, null before the first check"
    )
    checked_at: datetime | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    reason_code: str | None = None


class RemoteServerCreate(BaseModel):
    """Configuration submitted to register a new SSH-accessible remote server."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1, max_length=255)
    auth_type: SSHAuthType
    device_type: RemoteServerDeviceType
    ssh_secret: str = Field(min_length=1, description="Private key contents (key auth) or password (password auth)")
    ssh_key_passphrase: str | None = Field(
        default=None,
        min_length=1,
        description="Optional passphrase for a passphrase-protected private key (key auth only)",
    )

    @model_validator(mode="after")
    def validate_auth_material(self) -> "RemoteServerCreate":
        """Keep the passphrase scoped to key-based authentication."""
        if self.auth_type is SSHAuthType.PASSWORD and self.ssh_key_passphrase is not None:
            raise ValueError("ssh_key_passphrase is only valid when auth_type='key'")
        return self


class RemoteServerUpdate(BaseModel):
    """Mutable fields for a registered SSH remote server.

    Omitted fields are left unchanged. `ssh_secret`/`ssh_key_passphrase` are
    write-only: supplying one rotates the stored, encrypted value.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=255)
    host: str | None = Field(default=None, min_length=1, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = Field(default=None, min_length=1, max_length=255)
    auth_type: SSHAuthType | None = None
    device_type: RemoteServerDeviceType | None = None
    ssh_secret: str | None = Field(default=None, min_length=1, description="Rotate the stored secret")
    ssh_key_passphrase: str | None = Field(default=None, min_length=1, description="Rotate the stored passphrase")


class RemoteServer(BaseModel):
    """Public, sanitized view of a registered SSH remote server.

    Never carries `ssh_secret`, `ssh_key_passphrase`, or `host_key`.
    """

    id: UUID
    name: str
    host: str
    port: int
    username: str
    auth_type: SSHAuthType
    device_type: DeviceType
    last_check: LastCheckSummary = Field(default_factory=LastCheckSummary)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_serializer("id")
    def serialize_id(self, id: UUID, _info: Any) -> str:
        return str(id)


class RemoteServerInternal(RemoteServer):
    """Full persisted record, including confidential/internal fields.

    Used only by the repository/mapper and the SSH provisioning boundary.
    Never returned directly from a service's public API or an HTTP response,
    always convert with :meth:`to_public` first.
    """

    ssh_secret_encrypted: str = Field(
        description="Fernet ciphertext; never decrypted outside the provisioning boundary"
    )
    ssh_key_passphrase_encrypted: str | None = Field(
        default=None, description="Fernet ciphertext; never decrypted outside the provisioning boundary"
    )
    host_key: str | None = Field(
        default=None,
        description="Pinned public host key (TOFU); integrity data, not a secret, never serialized to API",
    )

    def to_public(self) -> RemoteServer:
        """Return the sanitized, API-safe view of this record."""
        return RemoteServer(
            **self.model_dump(exclude={"ssh_secret_encrypted", "ssh_key_passphrase_encrypted", "host_key"})
        )

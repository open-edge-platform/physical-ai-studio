from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from schemas.hardware import DeviceInfo, StorageInfo
from schemas.remote_server import SSH_HOST_ALIAS_PATTERN

HealthStatus = Literal["healthy", "degraded", "unreachable"]


class RemoteTrainerCreate(BaseModel):
    """Configuration for a direct remote trainer endpoint.

    ``ssh_host_alias`` is optional and, like `RemoteServerCreate`, names a
    ``Host`` stanza in the user's own ``~/.ssh/config`` rather than storing any
    key, password, or passphrase - Studio never persists SSH credentials for a
    direct trainer any more than it does for an SSH-provisioned one. When set,
    Studio keeps a standing SSH local-forward tunnel open for this trainer
    (``ssh_local_port`` on the studio host -> ``ssh_remote_port`` on the SSH
    host's own loopback interface), so ``url`` should point at that local
    port (typically ``http://127.0.0.1:<ssh_local_port>``).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    url: AnyHttpUrl
    ssh_host_alias: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=SSH_HOST_ALIAS_PATTERN,
        description="Name of a Host stanza in the user's SSH config, for an optional port-forward tunnel. Non-secret.",
    )
    ssh_remote_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port on the SSH host's loopback interface to forward to. Defaults to the trainer URL's port.",
    )
    ssh_local_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Loopback port on the studio host the tunnel binds to. Required when ssh_host_alias is set.",
    )

    @model_validator(mode="after")
    def _validate_ssh_tunnel(self) -> "RemoteTrainerCreate":
        """Require a coherent tunnel config, or none at all."""
        if self.ssh_host_alias is None:
            if self.ssh_remote_port is not None or self.ssh_local_port is not None:
                raise ValueError("ssh_remote_port and ssh_local_port require ssh_host_alias to be set")
            return self
        if self.ssh_local_port is None:
            raise ValueError("ssh_local_port is required when ssh_host_alias is set, so the tunnel binds a stable port")
        if self.ssh_remote_port is None:
            if self.url.port is None:
                raise ValueError("ssh_remote_port is required when the trainer URL has no explicit port")
            self.ssh_remote_port = self.url.port
        return self


class RemoteTrainerUpdate(BaseModel):
    """Mutable fields for a direct remote trainer endpoint.

    Cross-field tunnel consistency is only enforced on create: an update that
    only touches ``name`` must not be rejected for fields it never mentions.
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    url: AnyHttpUrl | None = None
    ssh_host_alias: str | None = Field(default=None, min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)
    ssh_remote_port: int | None = Field(default=None, ge=1, le=65535)
    ssh_local_port: int | None = Field(default=None, ge=1, le=65535)


class RemoteTrainer(RemoteTrainerCreate):
    """Persisted direct remote trainer endpoint."""

    id: UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RemoteTrainerHealth(BaseModel):
    """A sanitized, point-in-time health result for a configured trainer."""

    remote_trainer_id: UUID
    status: HealthStatus
    checked_at: datetime
    latency_ms: int | None = Field(default=None, ge=0)
    devices: list[DeviceInfo] = Field(default_factory=list)
    storage: StorageInfo | None = Field(
        default=None,
        description="Available storage on the trainer, when reported. Absence does not affect health status.",
    )
    reason_code: str | None = None

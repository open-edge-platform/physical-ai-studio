from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from schemas.hardware import DeviceInfo, StorageInfo
from schemas.remote_server import SSH_HOST_ALIAS_PATTERN

HealthStatus = Literal["healthy", "degraded", "unreachable", "starting"]


class RemoteTrainerConnectionMode(StrEnum):
    """How Studio reaches a configured remote trainer."""

    DIRECT = "direct"
    SSH = "ssh"


class ManualSshConnection(BaseModel):
    """Non-secret SSH connection fields entered directly by the user."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hostname: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    user: str | None = Field(default="ec2-user", max_length=255)
    identity_file: str | None = Field(
        default=None,
        max_length=4096,
        description="Path to a private key file. Studio never reads or stores its contents.",
    )


class RemoteTrainerCreate(BaseModel):
    """Configuration for a direct or managed SSH remote trainer.

    SSH aliases refer to ``Host`` entries in the user's ``~/.ssh/config``.
    Studio starts the managed container and forwards its host loopback port
    through a standing tunnel; the local URL is derived from ``ssh_local_port``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    connection_mode: RemoteTrainerConnectionMode = RemoteTrainerConnectionMode.DIRECT
    url: AnyHttpUrl | None = None
    ssh_host_alias: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=SSH_HOST_ALIAS_PATTERN,
        description="Name of a Host entry in the user's SSH config, for an optional port-forward tunnel. Non-secret.",
    )
    ssh_connection: ManualSshConnection | None = Field(
        default=None,
        description="Manual SSH connection fields for a tunnel which does not use an SSH config alias.",
    )
    ssh_remote_port: int | None = Field(
        default=8001,
        ge=1,
        le=65535,
        description="Port on the SSH host's loopback interface to forward to. Defaults to the trainer URL's port.",
    )
    ssh_local_port: int | None = Field(
        default=8001,
        ge=1,
        le=65535,
        description="Loopback port on the studio host the tunnel binds to. Required when ssh_host_alias is set.",
    )

    @model_validator(mode="after")
    def _validate_connection(self) -> Self:
        """Require fields coherent with the selected connection mode."""
        if self.connection_mode is RemoteTrainerConnectionMode.DIRECT:
            if self.url is None:
                raise ValueError("url is required for a direct connection")
            has_explicit_ssh_port = any(
                field in self.model_fields_set and getattr(self, field) is not None
                for field in ("ssh_remote_port", "ssh_local_port")
            )
            if self.ssh_host_alias is not None or self.ssh_connection is not None or has_explicit_ssh_port:
                raise ValueError("SSH fields require connection_mode='ssh'")
            self.ssh_remote_port = None
            self.ssh_local_port = None
            return self

        if self.ssh_host_alias is not None and self.ssh_connection is not None:
            raise ValueError("ssh_host_alias and ssh_connection are mutually exclusive")
        if self.ssh_host_alias is None and self.ssh_connection is None:
            raise ValueError("SSH mode requires either ssh_host_alias or ssh_connection")
        if self.ssh_local_port is None:
            raise ValueError("ssh_local_port is required when ssh_host_alias is set, so the tunnel binds a stable port")
        if self.ssh_remote_port is None:
            raise ValueError("ssh_remote_port is required for an SSH connection")
        self.url = AnyHttpUrl(f"http://127.0.0.1:{self.ssh_local_port}")
        return self


class RemoteTrainerUpdate(BaseModel):
    """Mutable fields for a remote trainer endpoint.

    Cross-field tunnel consistency is only enforced on create: an update that
    only touches ``name`` must not be rejected for fields it never mentions.
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    url: AnyHttpUrl | None = None
    connection_mode: RemoteTrainerConnectionMode | None = None
    ssh_host_alias: str | None = Field(default=None, min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)
    ssh_connection: ManualSshConnection | None = None
    ssh_remote_port: int | None = Field(default=None, ge=1, le=65535)
    ssh_local_port: int | None = Field(default=None, ge=1, le=65535)


class RemoteTrainer(RemoteTrainerCreate):
    """Persisted remote trainer endpoint."""

    id: UUID
    url: AnyHttpUrl  # pyrefly: ignore[bad-override-mutable-attribute]
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

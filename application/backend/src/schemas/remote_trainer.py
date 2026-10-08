from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, computed_field, model_validator

from schemas.hardware import DeviceInfo, StorageInfo
from schemas.remote_server import SSH_HOST_ALIAS_PATTERN

HealthStatus = Literal["healthy", "degraded", "unreachable", "starting"]


class RemoteTrainerConnectionMode(StrEnum):
    """How Studio reaches a configured remote trainer."""

    DIRECT = "direct"
    SSH = "ssh"
    AWS_BATCH = "aws_batch"


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


class DirectConnection(BaseModel):
    """A trainer service reachable at a URL Studio can open directly."""

    model_config = ConfigDict(str_strip_whitespace=True)

    connection_mode: Literal[RemoteTrainerConnectionMode.DIRECT] = RemoteTrainerConnectionMode.DIRECT
    url: AnyHttpUrl


class SshConnection(BaseModel):
    """A trainer container Studio starts on an SSH host and reaches through a tunnel.

    ``ssh_host_alias`` refers to a ``Host`` entry in the user's ``~/.ssh/config``;
    ``ssh_connection`` carries the same information entered by hand. Exactly one
    of the two must be set.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    connection_mode: Literal[RemoteTrainerConnectionMode.SSH] = RemoteTrainerConnectionMode.SSH
    ssh_host_alias: str | None = Field(default=None, min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)
    ssh_connection: ManualSshConnection | None = None
    ssh_remote_port: int = Field(
        default=8001, ge=1, le=65535, description="Loopback port on the SSH host the trainer container publishes."
    )
    ssh_local_port: int = Field(
        default=8001, ge=1, le=65535, description="Loopback port on the Studio host the tunnel binds to."
    )

    @model_validator(mode="after")
    def _exactly_one_host(self) -> Self:
        if self.ssh_host_alias is not None and self.ssh_connection is not None:
            raise ValueError("ssh_host_alias and ssh_connection are mutually exclusive")
        if self.ssh_host_alias is None and self.ssh_connection is None:
            raise ValueError("SSH mode requires either ssh_host_alias or ssh_connection")
        return self

    @property
    def host_name(self) -> str:
        """Human-readable host identifier for logs and UI."""
        if self.ssh_host_alias is not None:
            return self.ssh_host_alias
        assert self.ssh_connection is not None  # noqa: S101 - guaranteed by _exactly_one_host
        return self.ssh_connection.hostname

    @property
    def url(self) -> AnyHttpUrl:
        """Local end of the tunnel."""
        return AnyHttpUrl(f"http://127.0.0.1:{self.ssh_local_port}")


class AwsBatchTarget(BaseModel):
    """Batch job queue and job definition provisioned for one instance type."""

    model_config = ConfigDict(str_strip_whitespace=True)

    queue: str = Field(min_length=1, max_length=2048)
    job_definition: str = Field(min_length=1, max_length=2048)


class AwsBatchConnection(BaseModel):
    """AWS Batch resources from the ``aws-batch-trainer`` CloudFormation stack output."""

    model_config = ConfigDict(str_strip_whitespace=True)

    connection_mode: Literal[RemoteTrainerConnectionMode.AWS_BATCH] = RemoteTrainerConnectionMode.AWS_BATCH
    schema_version: Literal[1] = 1
    region: str = Field(min_length=1, max_length=64)
    studio_role_arn: str = Field(min_length=1, max_length=2048)
    bucket: str = Field(min_length=3, max_length=63)
    targets: dict[str, AwsBatchTarget] = Field(min_length=1, description="Instance type -> Batch resources.")


TrainerConnection = Annotated[
    DirectConnection | SshConnection | AwsBatchConnection,
    Field(discriminator="connection_mode"),
]


class RemoteTrainerCreate(BaseModel):
    """Configuration for a remote trainer; ``connection`` carries the mode-specific fields."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    connection: TrainerConnection


class RemoteTrainerUpdate(BaseModel):
    """Mutable fields for a remote trainer; ``connection`` is replaced as a whole."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    connection: TrainerConnection | None = None


class RemoteTrainer(RemoteTrainerCreate):
    """Persisted remote trainer endpoint."""

    id: UUID
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connection_mode(self) -> RemoteTrainerConnectionMode:
        return self.connection.connection_mode

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> AnyHttpUrl | None:
        """HTTP endpoint of the trainer service, when the mode has one."""
        if isinstance(self.connection, DirectConnection | SshConnection):
            return self.connection.url
        return None


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

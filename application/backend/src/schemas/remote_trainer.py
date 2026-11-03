import re
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from schemas.aws_auth import AwsCredentialSelection, AwsProfileName
from schemas.base import UTCDatetime
from schemas.hardware import DeviceInfo, StorageInfo
from schemas.operational_issue import OperationalIssue
from schemas.remote_server import SSH_HOST_ALIAS_PATTERN

HealthStatus = Literal["healthy", "degraded", "unreachable", "starting"]
AWS_REGION_PATTERN = r"[a-z]{2,}(-[a-z]+)+-\d"
AWS_BATCH_ARN_PREFIX = rf"arn:aws(?:-[a-z]+)*:batch:{AWS_REGION_PATTERN}:[0-9]{{12}}:"


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
    ssh_local_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Loopback port for the tunnel. Studio assigns an available port when omitted.",
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
    def url(self) -> AnyHttpUrl | None:
        """Local end of the tunnel."""
        if self.ssh_local_port is None:
            return None
        return AnyHttpUrl(f"http://127.0.0.1:{self.ssh_local_port}")


class AwsBatchTarget(BaseModel):
    """Batch job queue and job definition provisioned for one instance type."""

    model_config = ConfigDict(str_strip_whitespace=True)

    queue: str = Field(max_length=2048, pattern=rf"^{AWS_BATCH_ARN_PREFIX}job-queue/[a-zA-Z0-9][a-zA-Z0-9_-]{{0,127}}$")
    job_definition: str = Field(
        max_length=2048,
        pattern=rf"^{AWS_BATCH_ARN_PREFIX}job-definition/[a-zA-Z0-9_-]{{1,128}}(:[1-9][0-9]*)?$",
    )


class AwsBatchProviderConfiguration(AwsCredentialSelection):
    """Connection details used to load a provisioned AWS Batch configuration."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    configuration_uri: str = Field(
        max_length=4096,
        pattern=r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[^?#]+$",
        title="S3 configuration file URI",
    )


class CloudProviderField(BaseModel):
    """Form constraints exposed by a cloud provider's configuration schema."""

    name: str
    title: str
    min_length: int | None = Field(default=None, alias="minLength")
    max_length: int | None = Field(default=None, alias="maxLength")
    pattern: str | None = None


class CloudProviderError(BaseModel):
    """Sanitized provider configuration resolution failure."""

    detail: str
    code: str | None = None
    title: str | None = None
    command: str | None = None


class CloudProviderSchema(BaseModel):
    """Configuration fields for a selectable cloud provider."""

    id: Literal["aws"] = "aws"
    title: str = "AWS"
    fields: list[CloudProviderField]


class _AwsBatchResources(BaseModel):
    """Resource constraints shared by configuration documents and connections."""

    model_config = ConfigDict(str_strip_whitespace=True)

    region: str = Field(max_length=64, pattern=rf"^{AWS_REGION_PATTERN}$")
    studio_role_arn: str = Field(
        max_length=2048,
        pattern=r"^arn:aws(?:-[a-z]+)*:iam::[0-9]{12}:role/([\x21-\x7e]+/)?[a-zA-Z0-9_+=,.@-]{1,64}$",
    )
    bucket: str = Field(min_length=3, max_length=63, pattern=r"^[a-z0-9][a-z0-9.-]+[a-z0-9]$")
    targets: dict[str, AwsBatchTarget] = Field(min_length=1, description="Instance type -> Batch resources.")

    @field_validator("bucket")
    @classmethod
    def _valid_bucket_name(cls, value: str) -> str:
        if (
            ".." in value
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", value)
            or value.startswith(("xn--", "sthree-", "amzn-s3-demo-"))
            or value.endswith(("-s3alias", "--ol-s3", ".mrap", "--x-s3", "--table-s3"))
        ):
            raise ValueError("Invalid S3 bucket name")
        if value.endswith("-an") and not re.fullmatch(rf".+-[0-9]{{12}}-{AWS_REGION_PATTERN}-an", value):
            raise ValueError("Account-regional S3 bucket names require an account ID and region before -an")
        return value

    @model_validator(mode="after")
    def _batch_resources_match_region(self) -> Self:
        for instance_type, target in self.targets.items():
            for field, arn in (("queue", target.queue), ("job_definition", target.job_definition)):
                if arn.split(":", 5)[3] != self.region:
                    raise ValueError(f"targets.{instance_type}.{field} must be in region {self.region}")
        return self


class AwsBatchResourceConfiguration(_AwsBatchResources):
    """Deployment-independent resource document stored in S3."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]


class AwsBatchConnection(_AwsBatchResources):
    """Resolved AWS Batch connection and resources."""

    connection_mode: Literal[RemoteTrainerConnectionMode.AWS_BATCH] = RemoteTrainerConnectionMode.AWS_BATCH
    schema_version: Literal[1] = 1
    configuration_uri: str | None = None
    profile_name: AwsProfileName = "default"


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
    created_at: UTCDatetime | None = None
    updated_at: UTCDatetime | None = None

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
    checked_at: UTCDatetime
    latency_ms: int | None = Field(default=None, ge=0)
    devices: list[DeviceInfo] = Field(default_factory=list)
    storage: StorageInfo | None = Field(
        default=None,
        description="Available storage on the trainer, when reported. Absence does not affect health status.",
    )
    reason_code: str | None = None
    issues: list[OperationalIssue] | None = Field(
        default=None, description="Current issues; [] confirms recovery, null means the check was inconclusive."
    )

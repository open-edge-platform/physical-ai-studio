"""AWS credential selection and capability schemas."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from settings import get_settings

AwsProfileName = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f\[\]]+$")]


def is_robot_enrolled() -> bool:
    """Return True when a Greengrass robot configuration file is present on the backend host."""
    return get_settings().greengrass_config_path.is_file()


class AwsCredentialSelection(BaseModel):
    """Backend-local AWS profile selection (optional when a robot certificate is enrolled)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile_name: AwsProfileName | None = None

    @model_validator(mode="after")
    def _require_profile_unless_robot_enrolled(self) -> Self:
        if self.profile_name is None and not is_robot_enrolled():
            raise ValueError("profile_name is required when no robot certificate is configured on the host")
        return self


class AwsAuthCapabilities(BaseModel):
    """AWS authentication capabilities available on the Studio backend host."""

    profiles: list[str] = Field(default_factory=list)
    use_device_certificate: bool = False


class AwsIdentity(BaseModel):
    """Identity returned by a successful credential check."""

    account: str
    arn: str


class AwsAuthenticationIssue(BaseModel):
    """Recoverable loss of AWS credentials."""

    profile_name: str
    code: str
    title: str
    detail: str
    command: str | None = None

"""AWS named profile selection."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

AwsProfileName = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f\[\]]+$")]


class AwsCredentialSelection(BaseModel):
    """Explicit backend-local AWS profile."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile_name: AwsProfileName


class AwsAuthCapabilities(BaseModel):
    """AWS profiles available on the Studio backend host."""

    profiles: list[str] = Field(default_factory=list)


class AwsIdentity(BaseModel):
    """Identity returned by a successful credential check."""

    account: str
    arn: str


class AwsAuthenticationIssue(BaseModel):
    """Recoverable loss of a named profile's credentials."""

    profile_name: str
    code: str
    title: str
    detail: str
    command: str | None = None

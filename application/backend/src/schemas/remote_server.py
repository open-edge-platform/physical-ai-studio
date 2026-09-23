"""Non-secret SSH configuration schemas shared by remote trainers."""

from pydantic import BaseModel, ConfigDict, Field

SSH_HOST_ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$"


class ResolvedSshHost(BaseModel):
    """The non-secret connection target resolved from an SSH config alias."""

    alias: str
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = None
    found: bool = Field(description="False when the alias is absent or matches only a wildcard.")


class SshHostAliasOption(BaseModel):
    """A selectable SSH config alias."""

    alias: str
    hostname: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    user: str | None = None


class SshHostAliasCreate(BaseModel):
    """Fields for a non-secret ``Host`` entry in ``~/.ssh/config``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alias: str = Field(min_length=1, max_length=255, pattern=SSH_HOST_ALIAS_PATTERN)
    hostname: str = Field(min_length=1, max_length=255)
    port: int = Field(default=22, ge=1, le=65535)
    user: str | None = Field(default=None, max_length=255)
    identity_file: str | None = Field(default=None, max_length=4096)

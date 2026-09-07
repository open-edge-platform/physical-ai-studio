# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, JsonConfigSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


def get_default_storage_dir() -> Path:
    """Return the platform-appropriate directory for persistent app data."""
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/physicalai").expanduser()

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser() / "physicalai"
        return Path("~/AppData/Local/physicalai").expanduser()

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        xdg_data_home_path = Path(xdg_data_home).expanduser()
        if xdg_data_home_path.is_absolute():
            return xdg_data_home_path / "physicalai"

    return Path("~/.local/share/physicalai").expanduser()


def get_settings_file_path() -> Path:
    """Return the user-configurable settings file path."""
    override = os.environ.get("SETTINGS_FILE")
    if override:
        return Path(override).expanduser()
    return get_default_storage_dir() / "settings.json"


class TrainerClientSettings(BaseModel):
    """Client-side timeouts for talking to a remote trainer service."""

    request_timeout_s: float = Field(default=30.0)
    download_read_timeout_s: float = Field(default=120.0)
    stream_reconnect_max_s: float = Field(default=900.0)
    stream_reconnect_backoff_max_s: float = Field(default=30.0)


class HuggingFaceSettings(BaseModel):
    """Hugging Face credentials for authenticated training downloads."""

    hf_token: SecretStr | None = Field(default=None)

    @field_validator("hf_token", mode="before")
    @classmethod
    def empty_token_is_unset(cls, value: str | None) -> str | None:
        """Treat an empty form submission as removal of the stored token."""
        if isinstance(value, str) and not value.strip():
            return None
        return value


_USER_CONFIG_GROUPS: tuple[str, ...] = ("trainer", "huggingface")


class UserConfigSettingsSource(JsonConfigSettingsSource):
    """JSON settings source restricted to user-configurable groups."""

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = super().__call__()
        return {key: value for key, value in data.items() if key in _USER_CONFIG_GROUPS}


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Application
    app_name: str = "Physical AI Studio"
    version: str = "0.1.0"
    summary: str = "Physical AI Studio server"
    description: str = (
        "Physical AI Studio is a framework to train robots. It allows the user to create datasets, "
        "models and the run inference."
    )
    openapi_url: str = "/api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["dev", "prod"] = "dev"
    storage_dir: Path = Field(default_factory=get_default_storage_dir, alias="STORAGE_DIR")
    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    @field_validator("storage_dir", mode="before")
    @classmethod
    def expand_storage_dir(cls, value: Path | str) -> Path:
        """Expand user-provided storage directories like ~/.local/share."""
        return Path(value).expanduser()

    # Data import/upload safety (shared for dataset/model/project imports)
    data_import_max_uncompressed_bytes: int = Field(
        default=200 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MAX_UNCOMPRESSED_BYTES",
    )
    # Maximum raw upload size (Content-Length) accepted before any processing.
    # Default: 100 GiB - supports large dataset imports while still guarding abuse.
    data_import_max_upload_bytes: int = Field(
        default=100 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MAX_UPLOAD_BYTES",
    )
    # Minimum free bytes that must remain on the target filesystem after the
    # upload / extraction lands.  Default: 1 GiB headroom.
    data_import_min_free_bytes: int = Field(
        default=1 * 1024 * 1024 * 1024,
        alias="DATA_IMPORT_MIN_FREE_BYTES",
    )

    @property
    def datasets_dir(self) -> Path:
        """Storage directory for datasets."""
        return self.storage_dir / "datasets"

    @property
    def data_dir(self) -> Path:
        """Storage directory for application data."""
        return self.storage_dir / "data"

    @property
    def snapshot_dir(self) -> Path:
        """Storage directory for snapshots."""
        return self.storage_dir / "snapshots"

    @property
    def cache_dir(self) -> Path:
        """Storage directory for cache."""
        return self.storage_dir / "cache"

    @property
    def models_dir(self) -> Path:
        """Storage directory for models."""
        return self.storage_dir / "models"

    @property
    def robots_dir(self) -> Path:
        """Storage directory for robots."""
        return self.storage_dir / "robots"

    @property
    def log_dir(self) -> Path:
        """Storage directory for logs."""
        return self.storage_dir / "logs"

    # User-configurable client-side remote trainer timeouts.
    trainer: TrainerClientSettings = TrainerClientSettings()
    # User-configurable Hugging Face credentials.
    huggingface: HuggingFaceSettings = HuggingFaceSettings()

    # SSH-provisioned remote training
    # Master switch. Off by default: the feature has no authentication model,
    # so it must never be on for a deployment that is not a single-user
    # localhost workstation. See `core.security.ssh_network_exposure`, which
    # additionally fails this closed at startup if the backend is bound to a
    # non-loopback address regardless of this setting.
    ssh_remote_trainer_enabled: bool = Field(default=False, alias="SSH_REMOTE_TRAINER_ENABLED")
    # Path to the user's SSH client config. asyncssh parses it to resolve a saved
    # `ssh_host_alias` into a hostname, port, user, and identity; Studio never
    # reads key material out of it.
    ssh_config_path: Path = Field(default=Path("~/.ssh/config"), alias="SSH_CONFIG_PATH")
    # Path to the user's known_hosts. Host keys are verified against this file by
    # asyncssh, which fails closed on an unknown or changed key.
    ssh_known_hosts_path: Path = Field(default=Path("~/.ssh/known_hosts"), alias="SSH_KNOWN_HOSTS_PATH")
    # Overall budget for one SSH connect + auth. Bounds a save request.
    ssh_connect_timeout_s: float = Field(default=10.0, alias="SSH_CONNECT_TIMEOUT_S")
    # Per-command budget for the cheap Tier 1 probes (docker version, nvidia-smi).
    ssh_command_timeout_s: float = Field(default=15.0, alias="SSH_COMMAND_TIMEOUT_S")
    # Budget for `docker pull` of the (multi-gigabyte) trainer image.
    ssh_image_pull_timeout_s: float = Field(default=1800.0, alias="SSH_IMAGE_PULL_TIMEOUT_S")
    # Overall budget for a full Tier 1 preflight, so a save can never hang.
    ssh_preflight_timeout_s: float = Field(default=30.0, alias="SSH_PREFLIGHT_TIMEOUT_S")
    # Minimum time between preflight/status SSH connections to one server. Shared
    # by the status endpoint and the GPU-busy re-check so UI polling cannot
    # disrupt a running job or pile up connections.
    ssh_preflight_throttle_s: float = Field(default=5.0, alias="SSH_PREFLIGHT_THROTTLE_S")
    # Concurrent SSH connections allowed per server, across preflight and status.
    ssh_max_connections_per_server: int = Field(default=2, alias="SSH_MAX_CONNECTIONS_PER_SERVER")
    # SSH keepalive interval, so a dead tunnel is detected rather than hanging.
    ssh_keepalive_interval_s: float = Field(default=15.0, alias="SSH_KEEPALIVE_INTERVAL_S")
    ssh_keepalive_count_max: int = Field(default=3, alias="SSH_KEEPALIVE_COUNT_MAX")
    # Free disk a server must have at save time for the image plus a nominal job.
    # The actual dataset size is re-checked at provisioning time.
    ssh_min_free_disk_bytes: int = Field(
        default=50 * 1024 * 1024 * 1024,
        alias="SSH_MIN_FREE_DISK_BYTES",
    )
    # Maximum characters of streamed remote command output forwarded per line and
    # per message. Remote output is environment-influenced, not trusted text.
    ssh_output_max_line_chars: int = Field(default=512, alias="SSH_OUTPUT_MAX_LINE_CHARS")
    ssh_output_max_total_chars: int = Field(default=4096, alias="SSH_OUTPUT_MAX_TOTAL_CHARS")
    # Container registry hosting the trainer images resolved for SSH jobs.
    trainer_image_registry: str = Field(
        default="ghcr.io/open-edge-platform",
        alias="TRAINER_IMAGE_REGISTRY",
    )

    # --- SSH provisioning: GPU-busy wait ------------------------------------
    # Backoff between GPU-busy re-checks while a job waits `pending`.
    ssh_gpu_wait_initial_backoff_s: float = Field(default=5.0, alias="SSH_GPU_WAIT_INITIAL_BACKOFF_S")
    ssh_gpu_wait_max_backoff_s: float = Field(default=60.0, alias="SSH_GPU_WAIT_MAX_BACKOFF_S")
    # A job waiting this long for a busy GPU fails rather than waiting forever.
    ssh_gpu_wait_giveup_s: float = Field(default=1800.0, alias="SSH_GPU_WAIT_GIVEUP_S")

    # --- SSH provisioning: container lifecycle ------------------------------
    # `docker stop`'s grace period before SIGKILL, bounding teardown latency.
    ssh_container_stop_timeout_s: int = Field(default=30, alias="SSH_CONTAINER_STOP_TIMEOUT_S")
    # Budget for the container to report healthy after launch, before the job
    # fails rather than uploading a dataset to a trainer that never came up.
    ssh_readiness_timeout_s: float = Field(default=120.0, alias="SSH_READINESS_TIMEOUT_S")
    ssh_readiness_poll_interval_s: float = Field(default=2.0, alias="SSH_READINESS_POLL_INTERVAL_S")

    # --- SSH provisioning: tunnel reconnect ---------------------------------
    # Total time budget to reconnect a dropped tunnel and resume against the
    # still-running container before the job fails.
    ssh_tunnel_reconnect_budget_s: float = Field(default=300.0, alias="SSH_TUNNEL_RECONNECT_BUDGET_S")
    ssh_tunnel_reconnect_backoff_max_s: float = Field(default=15.0, alias="SSH_TUNNEL_RECONNECT_BACKOFF_MAX_S")

    # --- SSH provisioning: image signature verification ---------------------
    # Pinned to the Studio release workflow so `cosign verify` cannot be
    # satisfied by a signature from an unrelated identity/issuer.
    cosign_certificate_identity_regexp: str = Field(
        default=r"https://github\.com/open-edge-platform/physical-ai-studio/\.github/workflows/.+",
        alias="COSIGN_CERTIFICATE_IDENTITY_REGEXP",
    )
    cosign_oidc_issuer: str = Field(
        default="https://token.actions.githubusercontent.com",
        alias="COSIGN_OIDC_ISSUER",
    )
    # Fails closed by default: a remote trainer host without `cosign` installed
    # blocks the job rather than launching an unverified image. Set to `false`
    # only for hosts where installing `cosign` is not viable; a failed
    # `cosign verify` (as opposed to `cosign` being absent) still always blocks,
    # since that indicates a signature mismatch rather than missing tooling.
    ssh_require_cosign_verification: bool = Field(default=True, alias="SSH_REQUIRE_COSIGN_VERIFICATION")

    # --- SSH provisioning: library-version policy ---------------------------
    # Minimum `physicalai-train` version a trainer image must report. A job
    # policy (e.g. a specific model family) can require newer; see
    # `services.ssh.docker_ops.check_library_version`.
    ssh_min_library_version: str = Field(default="0.1.0", alias="SSH_MIN_LIBRARY_VERSION")

    @field_validator("ssh_config_path", "ssh_known_hosts_path", mode="before")
    @classmethod
    def expand_ssh_path(cls, value: Path | str) -> Path:
        """Expand `~` so the default resolves to the running user's SSH config."""
        return Path(value).expanduser()

    # Runtime sessions
    # Seconds a session keeps running with no client attached before it exits by
    # itself. A websocket reconnect after a page refresh completes well under 5s,
    # so 45 leaves roughly nine times the headroom for a slow reload or a brief
    # network drop; at the other end it bounds an abandoned, torque-holding arm to
    # under a minute. Labs on flaky networks want it longer, an unattended rig
    # wants it shorter.
    runtime_idle_timeout_s: float = Field(default=45.0, alias="RUNTIME_IDLE_TIMEOUT_S")

    # Server
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=7860, alias="PORT")

    # Database
    database_file: str = Field(default="physicalai.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    allow_unknown_db_revision: bool = Field(default=False, alias="ALLOW_UNKNOWN_DB_REVISION")

    # Alembic
    alembic_config_path: str = Field(default="src/alembic.ini", alias="ALEMBIC_CONFIG_PATH")
    alembic_script_location: str = Field(default="src/alembic", alias="ALEMBIC_SCRIPT_LOCATION")

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"sqlite+aiosqlite:///{self.data_dir / self.database_file}"

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL"""
        return f"sqlite:///{self.data_dir / self.database_file}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Resolve init kwargs, user JSON, environment, then .env."""
        return (
            init_settings,
            UserConfigSettingsSource(settings_cls, json_file=get_settings_file_path()),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def write_user_settings(data: dict[str, Any]) -> None:
    """Atomically persist allowed user-configurable settings."""
    path = get_settings_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = {key: value for key, value in data.items() if key in _USER_CONFIG_GROUPS and value is not None}
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix="settings.", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(filtered, tmp_file, indent=2, default=_plain_secret_str)
            tmp_file.write("\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _plain_secret_str(value: Any) -> Any:
    """Serialize SecretStr values as plaintext in the local settings file."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def load_user_settings_file() -> dict[str, Any]:
    """Return the raw user settings JSON, or an empty mapping."""
    path = get_settings_file_path()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def merge_user_settings(patch: dict[str, Any]) -> None:
    """Apply a partial patch without overwriting omitted fields."""
    current = load_user_settings_file()
    for group in _USER_CONFIG_GROUPS:
        if group not in patch:
            continue
        value = patch[group]
        if value is None:
            current.pop(group, None)
        elif isinstance(value, dict):
            current.setdefault(group, {}).update(value)
    write_user_settings(current)


def get_settings() -> Settings:
    """Return settings freshly resolved from their sources."""
    return Settings()

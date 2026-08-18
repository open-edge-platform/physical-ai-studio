# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Remote training
    # Seconds to wait for trainer HTTP requests (excludes long-poll/SSE streams).
    trainer_request_timeout_s: float = Field(default=30.0, alias="TRAINER_REQUEST_TIMEOUT_S")
    # Seconds to wait between chunks while streaming the model artifact. A stalled
    # transfer (e.g. a proxy holding the connection open) must fail instead of
    # hanging the job forever; this is a per-read gap, not a total transfer cap.
    trainer_download_read_timeout_s: float = Field(default=120.0, alias="TRAINER_DOWNLOAD_READ_TIMEOUT_S")
    # Stop reconnecting after this continuous trainer outage.
    trainer_stream_reconnect_max_s: float = Field(default=900.0, alias="TRAINER_STREAM_RECONNECT_MAX_S")
    # Upper bound on the exponential backoff between event-stream reconnect attempts.
    trainer_stream_reconnect_backoff_max_s: float = Field(default=30.0, alias="TRAINER_STREAM_RECONNECT_BACKOFF_MAX_S")

    # SSH-provisioned remote training
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

    @field_validator("ssh_config_path", "ssh_known_hosts_path", mode="before")
    @classmethod
    def expand_ssh_path(cls, value: Path | str) -> Path:
        """Expand `~` so the default resolves to the running user's SSH config."""
        return Path(value).expanduser()

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104 # nosec B104
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


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

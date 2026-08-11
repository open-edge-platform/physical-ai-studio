# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer service configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerSettings(BaseSettings):
    """Trainer service settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Working directory for snapshots, checkpoints, and model archives.
    storage_dir: Path = Field(
        default=Path("~/.local/share/physicalai-trainer").expanduser(), alias="TRAINER_STORAGE_DIR"
    )

    @field_validator("storage_dir", mode="before")
    @classmethod
    def expand_storage_dir(cls, value: Path | str) -> Path:
        """Expand user-provided storage directories like ~/.local/share.

        Field(default=...expanduser()) only expands the *default* value;
        a value actually supplied via TRAINER_STORAGE_DIR (env var or
        .env file) bypasses that and is parsed into a Path as-is, so a
        literal "~" segment was never expanded. Matches the equivalent
        validator on the studio Settings.storage_dir field.
        """
        return Path(value).expanduser()

    # Concurrency cap for the queue worker. Defaults to a single GPU job.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8, alias="TRAINER_MAX_CONCURRENT_JOBS")

    # nosec B104 - trainer is intended to be reachable from other machines on a
    # trusted local network.
    host: str = Field(default="0.0.0.0", alias="TRAINER_HOST")  # nosec B104 # noqa: S104
    port: int = Field(default=8001, alias="TRAINER_PORT")

    # HTTP-upload safety limits to prevent disk exhaustion.
    max_uncompressed_bytes: int = Field(
        default=200 * 1024 * 1024 * 1024,
        alias="TRAINER_MAX_UNCOMPRESSED_BYTES",
    )
    min_free_bytes: int = Field(
        default=1 * 1024 * 1024 * 1024,
        alias="TRAINER_MIN_FREE_BYTES",
    )

    @property
    def db_path(self) -> Path:
        """SQLite file backing the job queue."""
        return self.storage_dir / "trainer.db"

    @property
    def datasets_dir(self) -> Path:
        """Directory holding datasets uploaded over HTTP."""
        return self.storage_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        """Directory holding trained model outputs."""
        return self.storage_dir / "models"

    @property
    def archives_dir(self) -> Path:
        """Directory holding zipped model artifacts for download."""
        return self.storage_dir / "archives"


@lru_cache
def get_settings() -> TrainerSettings:
    """Return cached trainer settings."""
    return TrainerSettings()  # type: ignore[call-arg]

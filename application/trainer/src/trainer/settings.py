# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Trainer service configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerSettings(BaseSettings):
    """Trainer service settings sourced from the environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    # Working directory for snapshots, checkpoints, and model archives.
    storage_dir: Path = Field(default=Path("~/.local/share/physicalai-trainer").expanduser(), alias="STORAGE_DIR")
    # Concurrency cap for the queue worker. Defaults to a single GPU job.
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8, alias="TRAINER_MAX_CONCURRENT_JOBS")
    # Explicit accelerator override; auto-detected when unset.
    device: str | None = Field(default=None, alias="TRAINER_DEVICE")

    # nosec B104 - trainer is intended to be reachable from other machines on a
    # trusted local network.
    host: str = Field(default="0.0.0.0", alias="HOST")  # nosec B104 # noqa: S104
    port: int = Field(default=8001, alias="PORT")

    @property
    def db_path(self) -> Path:
        """SQLite file backing the job queue."""
        return self.storage_dir / "trainer.db"

    @property
    def snapshots_dir(self) -> Path:
        """Directory holding pulled dataset snapshots."""
        return self.storage_dir / "snapshots"

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

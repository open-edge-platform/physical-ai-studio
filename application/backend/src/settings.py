# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    storage_dir: Path = Field(default=Path("~/.cache/geti_action").expanduser(), alias="STORAGE_DIR")
    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    supported_backends: list[str] = ["torch"]

    @property
    def datasets_dir(self) -> Path:
        """Storage directory for datasets."""
        return self.storage_dir / "datasets"

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

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104 # nosec B104
    port: int = Field(default=7860, alias="PORT")

    # Database
    database_file: str = Field(default="geti_action.db", alias="DATABASE_FILE", description="Database filename")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = "src/alembic.ini"
    alembic_script_location: str = "src/alembic"

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

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
    app_name: str = "Geti Action"
    version: str = "0.1.0"
    summary: str = "Geti Action server"
    description: str = (
        "Geti Action is a framework to train robots."
        "It allows the user to create datasets, models and the run inference "
    )
    openapi_url: str = "/api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["dev", "prod"] = "dev"
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
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


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

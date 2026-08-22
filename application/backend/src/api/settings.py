# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-configurable application settings API."""

from fastapi import APIRouter
from pydantic import BaseModel

from settings import HuggingFaceSettings, Settings, TrainerClientSettings, get_settings, merge_user_settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettingsResponse(BaseModel):
    """Effective user-configurable settings, with secrets masked."""

    trainer: TrainerClientSettings
    huggingface: HuggingFaceSettings

    @classmethod
    def from_settings(cls, settings: Settings) -> "UserSettingsResponse":
        return cls(
            trainer=settings.trainer,
            huggingface=settings.huggingface,
        )


class SettingsUpdate(BaseModel):
    """Partial update for user-configurable application settings."""

    trainer: TrainerClientSettings | None = None
    huggingface: HuggingFaceSettings | None = None


@router.get("")
async def get_user_settings() -> UserSettingsResponse:
    """Get the effective user-configurable settings."""
    return UserSettingsResponse.from_settings(get_settings())


@router.patch("")
async def update_user_settings(update: SettingsUpdate) -> UserSettingsResponse:
    """Persist the supplied fields and return effective settings."""
    merge_user_settings(update.model_dump(exclude_unset=True))
    return UserSettingsResponse.from_settings(get_settings())

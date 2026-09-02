# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-configurable application settings API."""

from fastapi import APIRouter
from pydantic import BaseModel

from settings import (
    HuggingFaceSettings,
    Settings,
    SshProvisioningSettings,
    TrainerClientSettings,
    get_settings,
    merge_user_settings,
    ssh_patch_to_flat,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettingsResponse(BaseModel):
    """Effective user-configurable settings, with secrets masked."""

    trainer: TrainerClientSettings
    huggingface: HuggingFaceSettings
    ssh: SshProvisioningSettings

    @classmethod
    def from_settings(cls, settings: Settings) -> "UserSettingsResponse":
        return cls(
            trainer=settings.trainer,
            huggingface=settings.huggingface,
            ssh=settings.ssh,
        )


class SettingsUpdate(BaseModel):
    """Partial update for user-configurable application settings."""

    trainer: TrainerClientSettings | None = None
    huggingface: HuggingFaceSettings | None = None
    ssh: SshProvisioningSettings | None = None


@router.get("")
async def get_user_settings() -> UserSettingsResponse:
    """Get the effective user-configurable settings."""
    return UserSettingsResponse.from_settings(get_settings())


@router.patch("")
async def update_user_settings(update: SettingsUpdate) -> UserSettingsResponse:
    """Persist the supplied fields and return effective settings."""
    patch = update.model_dump(exclude_unset=True)
    if "ssh" in patch:
        # The SSH knobs are grouped for the API and the UI but stored flat,
        # because the services read them off `Settings` directly and each one
        # keeps its own documented `SSH_*` environment alias.
        patch.update(ssh_patch_to_flat(patch.pop("ssh")))
    merge_user_settings(patch)
    return UserSettingsResponse.from_settings(get_settings())

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-configurable application settings API."""

import asyncio

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from api.dependencies import SystemServiceDep
from settings import (
    HotkeySettings,
    HuggingFaceSettings,
    Settings,
    SshProvisioningSettings,
    TrainerClientSettings,
    VideoRecordingSettings,
    get_settings,
    merge_user_settings,
    ssh_patch_to_flat,
)

router = APIRouter(prefix="/api/settings", tags=["Settings"])


class UserSettingsResponse(BaseModel):
    """Effective user-configurable settings, with secrets masked."""

    video_recording: VideoRecordingSettings
    trainer: TrainerClientSettings
    huggingface: HuggingFaceSettings
    ssh: SshProvisioningSettings
    hotkeys: HotkeySettings

    @classmethod
    def from_settings(cls, settings: Settings) -> "UserSettingsResponse":
        return cls(
            video_recording=settings.video_recording,
            trainer=settings.trainer,
            huggingface=settings.huggingface,
            ssh=settings.ssh,
            hotkeys=settings.hotkeys,
        )


class SettingsUpdate(BaseModel):
    """Partial update for user-configurable application settings."""

    video_recording: VideoRecordingSettings | None = None
    trainer: TrainerClientSettings | None = None
    huggingface: HuggingFaceSettings | None = None
    ssh: SshProvisioningSettings | None = None
    hotkeys: HotkeySettings | None = None


@router.get("")
async def get_user_settings() -> UserSettingsResponse:
    """Get the effective user-configurable settings."""
    return UserSettingsResponse.from_settings(get_settings())


@router.patch("")
async def update_user_settings(update: SettingsUpdate, system_service: SystemServiceDep) -> UserSettingsResponse:
    """Persist the supplied fields and return effective settings."""
    patch = update.model_dump(exclude_unset=True)
    if video_recording := patch.get("video_recording"):
        preset_id = video_recording.get("preset")
        available_preset_ids = {
            preset.id for preset in (await asyncio.to_thread(system_service.get_video_recording_presets)).presets
        }
        if preset_id is not None and preset_id not in available_preset_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Video recording preset '{preset_id}' is unavailable on this backend.",
            )
    if "ssh" in patch:
        # The SSH knobs are grouped for the API and the UI but stored flat,
        # because the services read them off `Settings` directly and each one
        # keeps its own documented `SSH_*` environment alias.
        patch.update(ssh_patch_to_flat(patch.pop("ssh")))
    merge_user_settings(patch)
    return UserSettingsResponse.from_settings(get_settings())

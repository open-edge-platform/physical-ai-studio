# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""User-configurable application settings API."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from api.dependencies import HealthServiceDep
from core.restart import request_graceful_restart
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
async def update_user_settings(
    update: SettingsUpdate,
    background_tasks: BackgroundTasks,
    health_service: HealthServiceDep,
) -> UserSettingsResponse:
    """Persist the supplied fields and return effective settings.

    Toggling the SSH remote-trainer master switch (``ssh.enabled``) only takes
    effect after a full backend restart: `core.security.get_ssh_feature_availability`
    is cached for the life of the process and `app.state.ssh_feature_availability`
    is pinned once at startup. So, exactly when that field's effective value
    changes, this schedules the same graceful restart `POST /api/system/restart`
    uses - the save still completes and is reflected in the response; only the
    running process's behavior lags until the restart lands.
    """
    previously_enabled = get_settings().ssh_remote_trainer_enabled
    patch = update.model_dump(exclude_unset=True)
    if "ssh" in patch:
        # The SSH knobs are grouped for the API and the UI but stored flat,
        # because the services read them off `Settings` directly and each one
        # keeps its own documented `SSH_*` environment alias.
        patch.update(ssh_patch_to_flat(patch.pop("ssh")))
    merge_user_settings(patch)
    response = UserSettingsResponse.from_settings(get_settings())
    if response.ssh.enabled != previously_enabled:
        health_service.mark_plugin_restart_required()
        background_tasks.add_task(request_graceful_restart)
    return response

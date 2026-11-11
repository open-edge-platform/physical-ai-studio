# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for hardware discovery."""

import asyncio
import os
import signal
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel

from api.dependencies import HealthServiceDep, get_system_service
from build_info import get_build_info_async
from schemas.hardware import InferenceDeviceInfo, TrainingDevices
from schemas.video_recording import VideoRecordingPresets
from services.system_service import SystemService
from settings import get_settings

system_router = APIRouter(prefix="/api/system", tags=["System"])


class SystemInfo(BaseModel):
    """Version and source provenance of the running Studio application."""

    studio_version: str
    git_sha: str | None = None
    git_commit_date: str | None = None
    git_tag: str | None = None
    git_dirty: bool = False


@system_router.get("/info")
async def get_system_info() -> SystemInfo:
    """Return Studio version and the source revision used to build it."""
    build_info = await get_build_info_async()
    return SystemInfo(studio_version=get_settings().version, **build_info.__dict__)


def request_graceful_restart() -> None:
    """Send this process SIGTERM so the process supervisor restarts it.

    The FastAPI lifespan (`core.lifecycle.lifespan`) stops workers on
    receiving the signal, then re-executes the process
    (`core.lifecycle._restart_process`) if
    `HealthService.plugin_restart_required` was set beforehand. Callers must
    set that flag first - this function only requests the shutdown half of
    that sequence.
    """
    os.kill(os.getpid(), signal.SIGTERM)


@system_router.get("/devices/inference")
async def get_inference_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[InferenceDeviceInfo]:
    """Returns the list of available inference devices for OpenVINO and Torch."""
    return system_service.get_inference_devices()


@system_router.get("/devices/training")
async def get_training_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> TrainingDevices:
    """Returns the available training devices (CPU, Intel XPU, NVIDIA CUDA) and remote status.

    In remote training mode the devices reflect the remote trainer's hardware. If
    the trainer cannot be reached, ``remote_available`` is False and no devices
    are returned so the UI can block training instead of falling back to local CPU.
    """
    return await system_service.get_available_training_devices()


@system_router.get("/video-recording-presets")
async def get_video_recording_presets(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> VideoRecordingPresets:
    """Return recording presets verified by this backend's video stack."""
    return await asyncio.to_thread(system_service.get_video_recording_presets)


@system_router.post("/restart", status_code=status.HTTP_202_ACCEPTED)
async def restart_server(background_tasks: BackgroundTasks, health_service: HealthServiceDep) -> dict[str, str]:
    """Gracefully restart the server to activate plugin changes.

    The shutdown signal is sent after the response is flushed, allowing the
    FastAPI lifespan to stop workers before the process supervisor restarts it.
    """
    health_service.mark_plugin_restart_required()
    background_tasks.add_task(request_graceful_restart)
    return {"status": "restarting"}

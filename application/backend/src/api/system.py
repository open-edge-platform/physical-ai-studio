# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for hardware discovery."""

import os
import signal
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status

from api.dependencies import HealthServiceDep, get_system_service
from schemas.hardware import InferenceDeviceInfo, TrainingDevices
from services.system_service import SystemService

system_router = APIRouter(prefix="/api/system", tags=["System"])


def _stop_process() -> None:
    """Request graceful shutdown so the process supervisor can restart the server."""
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


@system_router.post("/restart", status_code=status.HTTP_202_ACCEPTED)
async def restart_server(background_tasks: BackgroundTasks, health_service: HealthServiceDep) -> dict[str, str]:
    """Gracefully restart the server to activate plugin changes.

    The shutdown signal is sent after the response is flushed, allowing the
    FastAPI lifespan to stop workers before the process supervisor restarts it.
    """
    health_service.mark_plugin_restart_required()
    background_tasks.add_task(_stop_process)
    return {"status": "restarting"}

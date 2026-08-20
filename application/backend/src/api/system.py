# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for hardware discovery."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_system_service
from schemas.hardware import InferenceDeviceInfo, TrainingDevices
from services.system_service import SystemService

system_router = APIRouter(prefix="/api/system", tags=["System"])


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

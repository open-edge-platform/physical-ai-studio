# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for hardware discovery."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_system_service
from schemas.hardware import DeviceInfo
from services.system_service import SystemService

system_router = APIRouter(prefix="/api/system", tags=["System"])


@system_router.get("/devices/training")
async def get_training_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfo]:
    """Returns the list of available training devices (CPU, Intel XPU, NVIDIA CUDA, Apple MPS)."""
    return system_service.get_training_devices()

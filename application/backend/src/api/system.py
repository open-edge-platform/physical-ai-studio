# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for hardware discovery."""

import os
import signal
import sys
import threading
import time
from typing import Annotated

from fastapi import APIRouter, Depends, status
from loguru import logger

from api.dependencies import get_system_service
from schemas.hardware import InferenceDeviceInfo, TrainingDevices
from services.system_service import SystemService

system_router = APIRouter(prefix="/api/system", tags=["System"])


def _restart_argv_candidates() -> list[list[str]]:
    """Build candidate argv lists for in-place process restart."""
    candidates: list[list[str]] = []

    orig_argv = list(getattr(sys, "orig_argv", []) or [])
    if orig_argv and orig_argv[0]:
        candidates.append(orig_argv)

    python_argv = [sys.executable, *sys.argv]
    if python_argv[0] and python_argv not in candidates:
        candidates.append(python_argv)

    return candidates


def _restart_process() -> bool:
    """Try to replace the current process image in-place."""
    for argv in _restart_argv_candidates():
        executable = argv[0]
        try:
            if os.path.sep in executable:
                os.execv(executable, argv)
            else:
                os.execvp(executable, argv)
            return True
        except OSError:
            logger.exception("Restart exec failed for argv={}", argv)
    return False


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
async def restart_server() -> dict[str, str]:
    """Gracefully restart the server to activate plugin changes.

    Schedules a delayed self-reexec so the response is flushed first. If reexec
    fails, falls back to SIGTERM so an external supervisor can restart it.
    """

    def _schedule_restart() -> None:
        time.sleep(1.0)
        if _restart_process():
            return
        logger.warning("Self-restart failed; falling back to SIGTERM")
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_schedule_restart, daemon=True).start()
    return {"status": "restarting"}

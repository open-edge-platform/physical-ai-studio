# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Hardware discovery for the trainer service.

Reports the compute devices this trainer can use for training so the studio
backend can surface the real (often GPU) hardware instead of the studio host's
local CPU. Mirrors the studio backend's device enumeration.
"""

from __future__ import annotations

from loguru import logger

from trainer.schemas import DeviceInfo
from trainer.settings import get_settings


def gpu_busy(device_type: str, index: int) -> bool | None:
    """Best-effort host-wide GPU memory check; None means telemetry unavailable."""
    try:
        import torch

        backend = torch.cuda if device_type == "cuda" else torch.xpu if device_type == "xpu" else None
        if backend is None:
            return None
        free, total = backend.mem_get_info(index)
        # Ignore this trainer's cached allocations (single-job mode trains in-process).
        # ponytail: Memory is a heuristic, not a reservation. Tune for desktop/driver overhead.
        return total - free - backend.memory_reserved(index) >= get_settings().gpu_busy_memory_mb * 1024 * 1024
    except Exception as exc:
        logger.warning("Cannot read {} GPU {} memory: {}", device_type, index, exc)
        return None


def get_training_devices() -> list[DeviceInfo]:
    """Enumerate Intel XPU and NVIDIA CUDA devices available for training."""
    devices: list[DeviceInfo] = []

    try:
        import torch
    except Exception as exc:
        logger.warning("torch unavailable; no accelerator devices reported: {}", exc)
        return devices

    try:
        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                xpu_props = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type="xpu",
                        name=xpu_props.name,
                        memory=xpu_props.total_memory,
                        index=device_idx,
                        busy=gpu_busy("xpu", device_idx),
                    ),
                )
    except Exception as exc:
        logger.warning("Failed to enumerate XPU devices: {}", exc)

    try:
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type="cuda",
                        name=cuda_props.name,
                        memory=cuda_props.total_memory,
                        index=device_idx,
                        busy=gpu_busy("cuda", device_idx),
                    ),
                )
    except Exception as exc:
        logger.warning("Failed to enumerate CUDA devices: {}", exc)

    return devices

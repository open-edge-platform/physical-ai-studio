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


def get_training_devices() -> list[DeviceInfo]:
    """Enumerate CPU, Intel XPU and NVIDIA CUDA devices available for training."""
    devices: list[DeviceInfo] = [DeviceInfo(type="cpu", name="CPU", memory=None, index=None)]

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch may be absent in lightweight installs
        logger.warning("torch unavailable; reporting CPU only: {}", exc)
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
                    ),
                )
    except Exception as exc:  # noqa: BLE001 - best-effort probe
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
                    ),
                )
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        logger.warning("Failed to enumerate CUDA devices: {}", exc)

    return devices

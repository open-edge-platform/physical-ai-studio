# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Service for querying system hardware information."""

import torch
from loguru import logger

from schemas.hardware import DeviceInfo, DeviceType


class SystemService:
    """Service to discover and report available compute hardware."""

    @staticmethod
    def get_training_devices() -> list[DeviceInfo]:
        """Get available compute devices for training.

        Enumerates CPU, Intel XPU, NVIDIA CUDA, and Apple MPS devices
        that PyTorch can use for model training.

        Returns:
            list[DeviceInfo]: Available training devices with name, type,
                memory (where available), and device index.
        """
        devices: list[DeviceInfo] = [
            DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None),
        ]

        # Intel XPU devices
        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.XPU,
                        name=props.name,
                        memory=props.total_memory,
                        index=device_idx,
                    ),
                )
                logger.debug("Detected XPU device {}: {} ({} bytes)", device_idx, props.name, props.total_memory)

        # NVIDIA CUDA devices
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.CUDA,
                        name=props.name,
                        memory=props.total_memory,
                        index=device_idx,
                    ),
                )
                logger.debug("Detected CUDA device {}: {} ({} bytes)", device_idx, props.name, props.total_memory)

        # Apple MPS
        if torch.mps.is_available():
            devices.append(
                DeviceInfo(type=DeviceType.MPS, name="MPS", memory=None, index=None),
            )
            logger.debug("Detected MPS device")

        return devices

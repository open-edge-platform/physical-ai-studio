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
                xpu_props = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.XPU,
                        name=xpu_props.name,
                        memory=xpu_props.total_memory,
                        index=device_idx,
                    ),
                )
                logger.debug(
                    "Detected XPU device {}: {} ({} bytes)",
                    device_idx,
                    xpu_props.name,
                    xpu_props.total_memory,
                )

        # NVIDIA CUDA devices
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.CUDA,
                        name=cuda_props.name,
                        memory=cuda_props.total_memory,
                        index=device_idx,
                    ),
                )
                logger.debug(
                    "Detected CUDA device {}: {} ({} bytes)",
                    device_idx,
                    cuda_props.name,
                    cuda_props.total_memory,
                )

        # Apple MPS
        if torch.mps.is_available():
            devices.append(
                DeviceInfo(type=DeviceType.MPS, name="MPS", memory=None, index=None),
            )
            logger.debug("Detected MPS device")

        return devices

    @classmethod
    def is_device_supported_for_training(cls, device_type: str) -> bool:
        """Check whether a device type is available for training.

        Args:
            device_type: Device type string, e.g. 'cpu', 'cuda', 'xpu'.

        Returns:
            True if at least one device of the given type is available.
        """
        device_type_lower = device_type.lower()
        return any(d.type == device_type_lower for d in cls.get_training_devices())

    @classmethod
    def supported_training_device_types(cls) -> list[str]:
        """Return the distinct device type strings available for training."""
        return sorted({d.type for d in cls.get_training_devices()})

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Service for querying system hardware information."""

import os
from importlib import import_module
from typing import Any

import torch
from loguru import logger

from schemas.hardware import DeviceInfo, DeviceType, InferenceBackend, InferenceDeviceInfo


class SystemService:
    """Service to discover and report available compute hardware."""

    @classmethod
    def get_inference_devices(cls) -> list[InferenceDeviceInfo]:
        """Get available backend-specific compute devices for inference.

        Returns:
            list[InferenceDeviceInfo]: Available inference devices with backend,
                backend-specific device identifier, name, memory, and index.
        """
        devices = cls._get_torch_inference_devices()
        devices.extend(cls._get_openvino_inference_devices())
        return devices

    @classmethod
    def _get_torch_inference_devices(cls) -> list[InferenceDeviceInfo]:
        """Get compute devices available to the Torch backend for inference."""
        system_memory = cls._get_system_memory()
        devices: list[InferenceDeviceInfo] = [
            InferenceDeviceInfo(
                backend=InferenceBackend.TORCH,
                device="cpu",
                type=DeviceType.CPU,
                name="CPU",
                memory=system_memory,
                index=None,
            ),
        ]

        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                xpu_props = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    InferenceDeviceInfo(
                        backend=InferenceBackend.TORCH,
                        device=f"xpu:{device_idx}",
                        type=DeviceType.XPU,
                        name=xpu_props.name,
                        memory=xpu_props.total_memory,
                        index=device_idx,
                    ),
                )

        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_props = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    InferenceDeviceInfo(
                        backend=InferenceBackend.TORCH,
                        device=f"cuda:{device_idx}",
                        type=DeviceType.CUDA,
                        name=cuda_props.name,
                        memory=cuda_props.total_memory,
                        index=device_idx,
                    ),
                )

        return devices

    @classmethod
    def _get_openvino_inference_devices(cls) -> list[InferenceDeviceInfo]:
        """Get compute devices available to the OpenVINO backend for inference."""
        try:
            openvino = import_module("openvino")
        except ImportError:
            logger.debug("OpenVINO is not installed; skipping OpenVINO inference devices.")
            return []

        core = openvino.Core()
        system_memory = cls._get_system_memory()
        devices: list[InferenceDeviceInfo] = [
            InferenceDeviceInfo(
                backend=InferenceBackend.OPENVINO,
                device="CPU",
                type=DeviceType.CPU,
                name="CPU",
                memory=system_memory,
                index=None,
            ),
        ]

        for device in core.available_devices:
            device_lower = device.lower()
            if device_lower.startswith("cpu"):
                continue

            name = cls._get_openvino_property(core, device, "FULL_DEVICE_NAME") or device
            if device_lower.startswith("npu"):
                devices.append(
                    InferenceDeviceInfo(
                        backend=InferenceBackend.OPENVINO,
                        device=device,
                        type=DeviceType.NPU,
                        name=str(name),
                        memory=cls._get_openvino_device_memory(core, device) or system_memory,
                        index=cls._get_openvino_device_index(core, device),
                    ),
                )
            elif device_lower.startswith("gpu"):
                devices.append(
                    InferenceDeviceInfo(
                        backend=InferenceBackend.OPENVINO,
                        device=device,
                        type=DeviceType.XPU,
                        name=str(name),
                        memory=cls._get_openvino_device_memory(core, device),
                        index=cls._get_openvino_device_index(core, device),
                    ),
                )
            else:
                logger.debug("Unsupported OpenVINO inference device '{}'; skipping.", device)

        return devices

    @staticmethod
    def _get_openvino_property(core: Any, device: str, property_name: str) -> Any | None:
        """Return an OpenVINO device property if supported by the runtime."""
        try:
            return core.get_property(device, property_name)
        except Exception as exc:
            logger.debug("Unable to read OpenVINO property '{}' for '{}': {}", property_name, device, exc)
            return None

    @classmethod
    def _get_openvino_device_memory(cls, core: Any, device: str) -> int | None:
        """Return OpenVINO device memory in bytes when the property is available."""
        memory = cls._get_openvino_property(core, device, "GPU_DEVICE_TOTAL_MEM_SIZE")
        return int(memory) if memory is not None else None

    @staticmethod
    def _get_system_memory() -> int | None:
        """Return total system memory in bytes when available."""
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_pages = os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            logger.debug("Unable to read total system memory.")
            return None
        return int(page_size * physical_pages)

    @classmethod
    def _get_openvino_device_index(cls, core: Any, device: str) -> int | None:
        """Return an OpenVINO device index when available."""
        device_id = cls._get_openvino_property(core, device, "DEVICE_ID")
        if device_id is not None:
            try:
                return int(device_id)
            except (TypeError, ValueError):
                logger.debug("OpenVINO DEVICE_ID for '{}' is not numeric: {}", device, device_id)

        _, separator, suffix = device.partition(".")
        if separator:
            try:
                return int(suffix)
            except ValueError:
                logger.debug("OpenVINO device suffix for '{}' is not numeric.", device)
        return None

    @staticmethod
    def get_training_devices() -> list[DeviceInfo]:
        """Get available compute devices for training.

        Enumerates CPU, Intel XPU and NVIDIA CUDA
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

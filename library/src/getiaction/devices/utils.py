# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Device detection and management utilities.

This module provides generic device detection that supports multiple accelerators:
- CUDA (NVIDIA GPUs)
- XPU (Intel GPUs)
- CPU (fallback)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "xpu", "cpu"]


def get_available_device() -> str:
    """Get the best available device for computation.

    Checks for available accelerators in order of preference:
    1. CUDA (NVIDIA GPUs)
    2. XPU (Intel GPUs)
    3. CPU (fallback)

    Returns:
        str: Device string ("cuda", "xpu", or "cpu").

    Example:
        >>> device = get_available_device()
        >>> print(device)
        'cuda'  # or 'xpu' or 'cpu' depending on hardware
    """
    if torch.cuda.is_available():
        logger.debug("CUDA device available")
        return "cuda"

    if torch.xpu.is_available():
        logger.debug("XPU device available")
        return "xpu"

    logger.debug("No GPU available, using CPU")
    return "cpu"


def get_device(device: str | None = None) -> torch.device:
    """Get a torch device, with automatic detection if not specified.

    Args:
        device: Device string (e.g., "cuda", "xpu", "cpu", "cuda:0", "auto").
            If None or "auto", automatically detects the best available device.

    Returns:
        torch.device: The resolved torch device.

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Explicit CUDA
        >>> device = get_device("xpu:0")  # Specific XPU device
    """
    if device is None or device == "auto":
        device = get_available_device()

    return torch.device(device)


def is_accelerator_available() -> bool:
    """Check if any GPU accelerator (CUDA or XPU) is available.

    Returns:
        bool: True if CUDA or XPU is available, False otherwise.

    Example:
        >>> if is_accelerator_available():
        ...     print("GPU acceleration available")
    """
    return get_available_device() != "cpu"


def get_device_count(device_type: DeviceType | None = None) -> int:
    """Get the number of available devices of a given type.

    Args:
        device_type: Type of device ("cuda", "xpu", "cpu").
            If None, returns count for the best available accelerator.

    Returns:
        int: Number of available devices. For CPU, returns 1.

    Example:
        >>> n_gpus = get_device_count("cuda")
        >>> print(f"Found {n_gpus} CUDA devices")
    """
    if device_type is None:
        device_type = get_available_device()  # type: ignore[assignment]

    if device_type == "cuda":
        return torch.cuda.device_count()
    if device_type == "xpu":
        return torch.xpu.device_count()
    return 1  # CPU


def get_device_name(device: str | torch.device | None = None) -> str:
    """Get the name of a device.

    Args:
        device: Device to get name for. If None, uses the best available device.

    Returns:
        str: Human-readable device name.

    Example:
        >>> print(get_device_name("cuda:0"))
        'NVIDIA GeForce RTX 4090'
    """
    if device is None:
        device = get_available_device()

    if isinstance(device, str):
        device = torch.device(device)

    device_obj: torch.device = device
    if device_obj.type == "cuda":
        idx = device_obj.index if device_obj.index is not None else 0
        return torch.cuda.get_device_name(idx)
    if device_obj.type == "xpu":
        idx = device_obj.index if device_obj.index is not None else 0
        return torch.xpu.get_device_name(idx)
    return "CPU"


def move_to_device(
    data: torch.Tensor | dict | list | tuple,
    device: str | torch.device,
) -> torch.Tensor | dict | list | tuple:
    """Recursively move tensors to a device.

    Args:
        data: Data to move (tensor, dict, list, or tuple containing tensors).
        device: Target device.

    Returns:
        Data with all tensors moved to the specified device.

    Example:
        >>> tensors = {"a": torch.randn(3), "b": [torch.randn(2), torch.randn(4)]}
        >>> tensors = move_to_device(tensors, "cuda")
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data

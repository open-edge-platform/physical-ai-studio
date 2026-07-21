# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for hardware and device information."""

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Enumeration of supported device types."""

    CPU = auto()
    XPU = auto()
    CUDA = auto()
    NPU = auto()


class InferenceBackend(StrEnum):
    """Enumeration of supported inference backends."""

    OPENVINO = auto()
    TORCH = auto()


class DeviceInfo(BaseModel):
    """Information about a compute device available for training."""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, cuda, npu)")
    name: str = Field(..., description="Human-readable device name")
    memory: int | None = Field(None, description="Total device memory in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")


class TrainingDevices(BaseModel):
    """Available training devices together with the active training mode.

    In remote mode the devices reflect the remote trainer's hardware. When the
    remote trainer cannot be reached, ``remote_available`` is False and
    ``devices`` is empty so callers can block training instead of silently
    falling back to local CPU-only training.
    """

    mode: Literal["local", "remote"] = Field(..., description="Active training mode (local or remote)")
    remote_available: bool = Field(
        ...,
        description="Whether the remote trainer is reachable. Always True in local mode.",
    )
    devices: list[DeviceInfo] = Field(default_factory=list, description="Available training devices")


class InferenceDevice(BaseModel):
    """Selected backend-specific inference device."""

    backend: InferenceBackend = Field(..., description="Inference backend (openvino, torch)")
    device: str = Field(..., description="Backend-specific device identifier")


class InferenceDeviceInfo(DeviceInfo):
    """Information about a backend-specific compute device available for inference."""

    backend: InferenceBackend = Field(..., description="Inference backend (openvino, torch)")
    device: str = Field(..., description="Backend-specific device identifier")

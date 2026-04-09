# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schemas for hardware and device information."""

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Enumeration of supported device types."""

    CPU = auto()
    XPU = auto()
    CUDA = auto()
    MPS = auto()
    NPU = auto()


class DeviceInfo(BaseModel):
    """Information about a compute device available for training."""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, cuda, mps, npu)")
    name: str = Field(..., description="Human-readable device name")
    memory: int | None = Field(None, description="Total device memory in bytes (null for CPU/MPS)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU/MPS)")

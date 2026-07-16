# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Request/response models for the trainer API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, Field, field_validator

_SUPPORTED_POLICIES = frozenset({"act", "pi0", "pi05", "smolvla"})


class DatasetTransfer(StrEnum):
    """How the dataset snapshot reaches the trainer."""

    # ZIP streamed directly to the trainer over HTTP. This is the only
    # supported transfer mode; datasets are never pulled from HuggingFace.
    HTTP = "http"


class TrainerJobStatus(StrEnum):
    """Lifecycle states for a trainer job."""

    # Job accepted, waiting for the dataset ZIP upload.
    AWAITING_DATASET = "awaiting_dataset"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class DeviceInfo(BaseModel):
    """Information about a compute device available on the trainer for training.

    Mirrors the studio backend's ``DeviceInfo`` schema so the studio can ingest
    the trainer's hardware report without translation.
    """

    type: str = Field(..., description="Device type (cpu, xpu, cuda)")
    name: str = Field(..., description="Human-readable device name")
    memory: int | None = Field(default=None, description="Total device memory in bytes (null for CPU)")
    index: int | None = Field(default=None, description="Device index among those of the same type (null for CPU)")


class SubmitJobRequest(BaseModel):
    """Job submission payload sent by the studio backend."""

    # Full TrainJobPayload as serialized by the client. Only training-relevant
    # fields are read server-side; the client device selection is ignored.
    payload: dict[str, Any]
    policy: str = Field(..., description="Policy name to train")
    dataset_transfer: DatasetTransfer = Field(
        default=DatasetTransfer.HTTP,
        description="How the dataset reaches the trainer (http upload)",
    )

    @field_validator("policy")
    @classmethod
    def _validate_policy(cls, value: str) -> str:
        if value not in _SUPPORTED_POLICIES:
            msg = f"Unsupported policy {value!r}"
            raise ValueError(msg)
        return value


class SubmitJobResponse(BaseModel):
    """Response returned after enqueueing a job."""

    remote_job_id: UUID
    status: TrainerJobStatus


class JobState(BaseModel):
    """Current state of a trainer job."""

    remote_job_id: UUID
    status: TrainerJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    extra_info: dict[str, Any] | None = None


class CancelResponse(BaseModel):
    """Status reported after a cancellation request."""

    remote_job_id: UUID
    status: TrainerJobStatus

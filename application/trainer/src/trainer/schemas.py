# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Request/response models for the trainer API."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Concrete 40-char hex commit SHA. Branch names / "main" are rejected so the
# server always pulls a pinned, immutable revision.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Conservative HuggingFace repo id: optional single namespace + repo name.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")

_SUPPORTED_POLICIES = frozenset({"act", "pi0", "pi05", "smolvla"})


class DatasetTransfer(StrEnum):
    """How the dataset snapshot reaches the trainer."""

    # ZIP streamed directly to the trainer over HTTP.
    HTTP = "http"
    # Pulled from an ephemeral private HuggingFace dataset repo.
    HF = "hf"


class TrainerJobStatus(StrEnum):
    """Lifecycle states for a trainer job."""

    # Job accepted, waiting for the dataset ZIP upload (http transfer only).
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
        description="How the dataset reaches the trainer (http upload or hf pull)",
    )
    # Required only for hf transfer; unused (and rejected) for http transfer.
    repo_id: str | None = Field(default=None, description="Ephemeral private HF dataset repo holding the snapshot")
    revision: str | None = Field(default=None, description="Pinned commit SHA of the snapshot repo")

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, value: str | None) -> str | None:
        if value is not None and not _REPO_ID_RE.fullmatch(value):
            msg = f"Invalid repo_id: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str | None) -> str | None:
        if value is not None and not _SHA_RE.fullmatch(value):
            msg = "revision must be a 40-character commit SHA"
            raise ValueError(msg)
        return value

    @field_validator("policy")
    @classmethod
    def _validate_policy(cls, value: str) -> str:
        if value not in _SUPPORTED_POLICIES:
            msg = f"Unsupported policy {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_transfer(self) -> SubmitJobRequest:
        """Enforce that repo fields match the chosen transfer mode."""
        if self.dataset_transfer == DatasetTransfer.HF:
            if not self.repo_id or not self.revision:
                msg = "hf transfer requires both repo_id and revision"
                raise ValueError(msg)
        elif self.repo_id is not None or self.revision is not None:
            msg = "http transfer must not set repo_id or revision"
            raise ValueError(msg)
        return self


class SubmitJobResponse(BaseModel):
    """Response returned after enqueueing a job."""

    remote_job_id: str
    status: TrainerJobStatus


class JobState(BaseModel):
    """Current state of a trainer job."""

    remote_job_id: str
    status: TrainerJobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: str | None = None
    extra_info: dict[str, Any] | None = None


class CancelResponse(BaseModel):
    """Status reported after a cancellation request."""

    remote_job_id: str
    status: TrainerJobStatus

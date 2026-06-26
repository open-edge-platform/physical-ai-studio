# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Request/response models for the trainer API."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Concrete 40-char hex commit SHA. Branch names / "main" are rejected so the
# server always pulls a pinned, immutable revision.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Conservative HuggingFace repo id: optional single namespace + repo name.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$")

_SUPPORTED_POLICIES = frozenset({"act", "pi0", "pi05", "smolvla"})


class TrainerJobStatus(StrEnum):
    """Lifecycle states for a trainer job."""

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
    repo_id: str = Field(..., description="Ephemeral private HF dataset repo holding the snapshot")
    revision: str = Field(..., description="Pinned commit SHA of the snapshot repo")
    policy: str = Field(..., description="Policy name to train")

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, value: str) -> str:
        if not _REPO_ID_RE.fullmatch(value):
            msg = f"Invalid repo_id: {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(value):
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

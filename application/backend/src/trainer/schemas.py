# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Request/response models for the trainer API."""

from __future__ import annotations

from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import UUID  # noqa: TC003

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from training import TrainingJobSpec

_SUPPORTED_POLICIES = frozenset({"act", "pi0", "pi05", "smolvla"})
_DEFAULT_PROTOCOL_VERSION = 1


def _installed_library_version() -> str:
    """Return the installed `physicalai-train` version, or "unknown".

    Read directly from installed package metadata rather than a Docker-build
    ARG, so the reported version can never drift from what is actually
    installed in this image.
    """
    try:
        return version("physicalai-train")
    except PackageNotFoundError:
        return "unknown"


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


class HealthInfo(BaseSettings):
    """Non-sensitive metadata used to verify trainer image compatibility.

    Sourced directly from the environment so the health endpoint doesn't need
    to read ``os.environ`` itself; fields fall back to safe defaults so a
    malformed environment never causes a 500 on liveness checks.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    status: str = Field(default="healthy", description="Service liveness status")
    protocol_version: int = Field(
        default=_DEFAULT_PROTOCOL_VERSION,
        alias="TRAINER_API_PROTOCOL_VERSION",
        description="Trainer API protocol version",
    )
    device_type: str = Field(
        default="unknown",
        alias="TRAINER_DEVICE_TYPE",
        description="Hardware target baked into this image",
    )
    build_revision: str = Field(
        default="unknown",
        alias="TRAINER_BUILD_REVISION",
        description="Source revision used to build the image",
    )
    build_date: str = Field(default="unknown", alias="TRAINER_BUILD_DATE", description="Image build timestamp")
    application_version: str = Field(
        default="unknown",
        alias="TRAINER_APPLICATION_VERSION",
        description="Physical AI Studio application version",
    )
    library_version: str = Field(
        default_factory=_installed_library_version,
        description=(
            "Installed physicalai-train version. Read from package metadata rather than an "
            "environment variable, so it always reflects what is actually installed."
        ),
    )

    @field_validator("protocol_version", mode="before")
    @classmethod
    def _coerce_protocol_version(cls, value: object) -> object:
        """Fall back to the default on an unparseable protocol version.

        Keeps /health resilient to a malformed environment so liveness
        checks never 500 on a bad build/deploy configuration.
        """
        if isinstance(value, int) or value is None:
            return value
        try:
            return int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            logger.warning(
                "Invalid TRAINER_API_PROTOCOL_VERSION={!r}; falling back to {}",
                value,
                _DEFAULT_PROTOCOL_VERSION,
            )
            return _DEFAULT_PROTOCOL_VERSION


class StorageInfo(BaseModel):
    """Disk usage for the volume backing the trainer's storage directory.

    Mirrors the studio backend's ``StorageInfo`` schema so the studio can
    ingest the trainer's storage report without translation.
    """

    total_bytes: int = Field(..., ge=0, description="Total capacity of the storage volume in bytes")
    free_bytes: int = Field(..., ge=0, description="Free space available on the storage volume in bytes")


class SubmitJobRequest(BaseModel):
    """Job submission payload sent by the studio backend.

    ``spec`` is the same :class:`~training.TrainingJobSpec` the studio builds
    for an in-process run, so a remote job trains from an identical
    configuration instead of from a re-parsed dict of loosely typed fields.
    """

    spec: TrainingJobSpec = Field(..., description="What to train (policy, steps, batch size, precision, device)")
    dataset_transfer: DatasetTransfer = Field(
        default=DatasetTransfer.HTTP,
        description="How the dataset reaches the trainer (http upload)",
    )

    @field_validator("spec")
    @classmethod
    def _validate_policy(cls, value: TrainingJobSpec) -> TrainingJobSpec:
        """Reject an unsupported policy at submission time.

        ``TrainingJobSpec.policy`` is a free-form string so new policies can be
        added without a schema change. Checking it here turns an unknown
        policy into a 422 on POST /jobs rather than a job that fails minutes
        later when the policy class is constructed.
        """
        if value.policy not in _SUPPORTED_POLICIES:
            msg = f"Unsupported policy {value.policy!r}"
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

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Batch job layout, as injected into the container by the job definition."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BatchSettings(BaseSettings):
    """Object names under the job's storage prefix and reporting cadence."""

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    bucket: str = Field(alias="PHYSICALAI_BATCH_BUCKET")
    job_prefix: str = Field(alias="PHYSICALAI_BATCH_JOB_PREFIX", min_length=1)
    spec_key: str = Field(default="spec.json", alias="PHYSICALAI_BATCH_SPEC_KEY")
    dataset_key: str = Field(default="dataset.zip", alias="PHYSICALAI_BATCH_DATASET_KEY")
    status_key: str = Field(default="status.json", alias="PHYSICALAI_BATCH_STATUS_KEY")
    artifact_key: str = Field(default="artifact.zip", alias="PHYSICALAI_BATCH_ARTIFACT_KEY")
    status_interval_s: float = Field(default=5.0, ge=0.0, alias="PHYSICALAI_BATCH_STATUS_INTERVAL_S")

    @property
    def job_id(self) -> str:
        """Last path segment of the prefix, used to name local work directories."""
        return self.job_prefix.rstrip("/").rsplit("/", 1)[-1]

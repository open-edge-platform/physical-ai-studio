# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Amazon S3 job storage for AWS Batch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from training import TrainingJobSpec

if TYPE_CHECKING:
    from pathlib import Path

    from trainer.batch.settings import BatchSettings
    from trainer.batch.status import BatchJobStatus


def _default_client() -> Any:
    import boto3

    return boto3.client("s3")


class S3JobStorage:
    """Job storage rooted at ``s3://{bucket}/{job_prefix}/``."""

    def __init__(self, settings: BatchSettings, client: Any | None = None) -> None:
        self._client: Any = client if client is not None else _default_client()
        self._bucket = settings.bucket
        self._prefix = settings.job_prefix.strip("/")
        self._settings = settings

    def _key(self, name: str) -> str:
        return f"{self._prefix}/{name}"

    def fetch_spec(self) -> TrainingJobSpec:
        body = self._client.get_object(Bucket=self._bucket, Key=self._key(self._settings.spec_key))["Body"].read()
        return TrainingJobSpec.model_validate_json(body)

    def fetch_dataset(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, self._key(self._settings.dataset_key), str(dest))

    def put_status(self, status: BatchJobStatus) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key(self._settings.status_key),
            Body=status.model_dump_json().encode(),
            ContentType="application/json",
        )

    def put_artifact(self, archive: Path) -> None:
        self._client.upload_file(str(archive), self._bucket, self._key(self._settings.artifact_key))

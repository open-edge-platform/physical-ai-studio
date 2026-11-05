# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.job import get_model_job_metrics
from exceptions import AwsFeatureDisabledError
from schemas.job import RemoteTrainJobPayload, TrainJob


@pytest.mark.parametrize("url", ["https://trainer.test", "http://trainer.test", "aws-batch://us-east-1/jobs-bucket"])
async def test_remote_metrics_select_the_trainer_transport(url: str, tmp_path: Path, monkeypatch) -> None:
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        remote_trainer_id=uuid4(),
        remote_trainer_url=url,
        remote_job_id=uuid4(),
        policy="act",
        model_name="model",
    )
    job = MagicMock(spec=TrainJob, payload=payload, id=uuid4())
    jobs = MagicMock(get_job_by_id=AsyncMock(return_value=job))
    metrics = MagicMock(get_model_job_metrics_path=AsyncMock(return_value=tmp_path / "missing.csv"))
    response = MagicMock()
    monkeypatch.setattr("api.job.EventSourceResponse", response)

    await get_model_job_metrics(uuid4(), jobs, metrics)

    if url.startswith(("http://", "https://")):
        metrics.tail_remote_csv_file.assert_called_once_with(url, str(payload.remote_job_id))
        metrics.empty_metrics_stream.assert_not_called()
        response.assert_called_once_with(metrics.tail_remote_csv_file.return_value)
    else:
        metrics.tail_remote_csv_file.assert_not_called()
        metrics.tail_batch_csv_file.assert_called_once_with(url, str(job.id))
        metrics.empty_metrics_stream.assert_not_called()
        response.assert_called_once_with(metrics.tail_batch_csv_file.return_value)


async def test_batch_metrics_reject_exposure_before_starting_stream(tmp_path: Path) -> None:
    job = TrainJob(
        project_id=uuid4(),
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            remote_trainer_id=uuid4(),
            remote_trainer_url="aws-batch://us-east-1/jobs-bucket",
            policy="act",
            model_name="model",
        ),
    )
    jobs = MagicMock(get_job_by_id=AsyncMock(return_value=job))
    metrics = MagicMock(get_model_job_metrics_path=AsyncMock(return_value=tmp_path / "missing.csv"))
    with (
        patch("api.job.require_aws_feature_active", side_effect=AwsFeatureDisabledError),
        pytest.raises(AwsFeatureDisabledError),
    ):
        await get_model_job_metrics(job.id, jobs, metrics)
    metrics.tail_batch_csv_file.assert_not_called()

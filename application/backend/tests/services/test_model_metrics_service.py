# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError, SSOTokenLoadError

from schemas.remote_trainer import AwsBatchConnection
from services.aws_batch import aws_batch_trainer_url
from services.model_metrics_service import ModelMetricsService


async def test_batch_metrics_wait_for_s3_and_emit_only_new_complete_rows(monkeypatch) -> None:
    connection = AwsBatchConnection(
        region="us-east-1",
        bucket="job-bucket",
        studio_role_arn="arn:aws:iam::123456789012:role/studio",
        targets={
            "g4dn.2xlarge": {
                "queue": "arn:aws:batch:us-east-1:123456789012:job-queue/queue",
                "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/definition:1",
            }
        },
    )
    client = MagicMock()
    first = BytesIO(b"step,train/loss\n0,2.5\n1,")
    unchanged = BytesIO(b"step,train/loss\n0,2.5\n")
    second = BytesIO(b"step,train/loss\n0,2.5\n1,1.25\n")
    client.get_object.side_effect = [
        ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject"),
        {"Body": first},
        {"Body": unchanged},
        {"Body": second},
    ]
    session = MagicMock()
    session.client.return_value = client
    make_session = MagicMock(return_value=session)
    monkeypatch.setattr("services.aws_batch._make_session", make_session)
    monkeypatch.setattr("services.model_metrics_service.asyncio.sleep", AsyncMock())
    stream = ModelMetricsService.tail_batch_csv_file(aws_batch_trainer_url(connection), "studio-job-id")
    try:
        initial = await anext(stream)
        assert initial.event == "issues"
        assert initial.data == "[]"
        assert json.loads((await anext(stream)).data) == {"step": 0, "train_loss": 2.5}
        assert json.loads((await anext(stream)).data) == {"step": 1, "train_loss": 1.25}
    finally:
        await stream.aclose()

    make_session.assert_called_once_with(connection)
    client.get_object.assert_called_with(Bucket="job-bucket", Key="jobs/studio-job-id/metrics.csv")
    assert first.closed and unchanged.closed and second.closed


async def test_local_metrics_snapshot_clears_issues_left_by_a_remote_stream(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("step,train/loss\n1,2.5\n")
    stream = ModelMetricsService.tail_csv_file(path)
    try:
        snapshot = await anext(stream)
        assert snapshot.event == "issues"
        assert snapshot.data == "[]"
        assert json.loads((await anext(stream)).data) == {"step": 1, "train_loss": 2.5}
    finally:
        await stream.aclose()


@pytest.mark.parametrize("missing_code", [None, "NoSuchKey", "404"])
async def test_batch_metrics_emit_authentication_and_resume_after_renewal(monkeypatch, missing_code) -> None:
    connection = AwsBatchConnection(
        region="us-east-1",
        profile_name="studio",
        bucket="job-bucket",
        studio_role_arn="arn:aws:iam::123456789012:role/studio",
        targets={
            "gpu": {
                "queue": "arn:aws:batch:us-east-1:123456789012:job-queue/queue",
                "job_definition": "arn:aws:batch:us-east-1:123456789012:job-definition/definition:1",
            }
        },
    )
    session = MagicMock()
    client = session.client.return_value
    client.get_object.side_effect = [
        *([ClientError({"Error": {"Code": missing_code}}, "GetObject")] if missing_code else []),
        {"Body": BytesIO(b"step,train/loss\n1,2.5\n")},
    ]
    monkeypatch.setattr(
        "services.aws_batch._make_session",
        MagicMock(
            side_effect=[
                SSOTokenLoadError(error_msg="expired"),
                session,
            ]
        ),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("services.model_metrics_service.asyncio.sleep", sleep)
    stream = ModelMetricsService.tail_batch_csv_file(aws_batch_trainer_url(connection), "job")
    try:
        issue = await anext(stream)
        assert issue.event == "issues"
        assert json.loads(issue.data)[0]["copy_text"] == "aws sso login --profile=studio"
        recovery = await anext(stream)
        assert recovery.event == "issues"
        assert recovery.data == "[]"
        assert client.get_object.call_count == 1
        sleep.assert_awaited_once_with(5)
        assert json.loads((await anext(stream)).data) == {"step": 1, "train_loss": 2.5}
    finally:
        await stream.aclose()

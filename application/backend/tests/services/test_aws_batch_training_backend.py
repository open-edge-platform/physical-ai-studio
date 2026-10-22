"""Exercise the Batch execution path with mocked AWS clients."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call
from uuid import uuid4
from zipfile import ZipFile

import pytest
import yaml
from botocore.exceptions import ClientError, SSOTokenLoadError
from pydantic import SecretStr

from schemas.job import RemoteTrainJobPayload, TrainJobPayloadAdapter
from schemas.remote_trainer import AwsBatchConnection, RemoteTrainer
from services.aws_batch import POLICY_INSTANCE_TYPES, aws_batch_connection_from_url, aws_batch_trainer_url
from services.remote_trainer_service import RemoteTrainerService
from services.training_backends import get_training_backend
from services.training_backends.aws_batch import AwsBatchTrainingBackend
from services.training_backends.base import TrainingCanceledError, TrainingContext, TrainingSuspendedError
from services.training_targets.remote import RemoteTrainingTargetHandler
from trainer.batch.status import BatchJobStatus
from training import TrainingJobSpec

MODULE = "services.training_backends.aws_batch"


@pytest.fixture
def connection() -> AwsBatchConnection:
    return AwsBatchConnection(
        region="us-east-1",
        studio_role_arn="arn:aws:iam::123456789012:role/studio",
        bucket="jobs-bucket",
        targets={
            instance_type: {
                "queue": f"arn:aws:batch:us-east-1:123456789012:job-queue/{size}-queue",
                "job_definition": f"arn:aws:batch:us-east-1:123456789012:job-definition/{size}-definition",
            }
            for instance_type, size in [("g4dn.2xlarge", "small"), ("g6e.2xlarge", "medium"), ("g7e.8xlarge", "large")]
        },
    )


@pytest.fixture
def context(tmp_path: Path, connection: AwsBatchConnection) -> TrainingContext:
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        remote_trainer_id=uuid4(),
        policy="act",
        model_name="model",
        remote_trainer_url=aws_batch_trainer_url(connection),
    )
    return TrainingContext(
        job=MagicMock(id=uuid4(), progress=0),
        model=MagicMock(policy="act"),
        snapshot=MagicMock(path=str(tmp_path / "dataset")),
        payload=payload,
        base_model=None,
        output_dir=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        progress=MagicMock(),
        should_stop=lambda: False,
        on_remote_job_id=AsyncMock(),
    )


@pytest.fixture
def aws(monkeypatch):
    batch, s3 = MagicMock(), MagicMock()
    session = MagicMock()
    session.client.side_effect = {"batch": batch, "s3": s3}.__getitem__
    monkeypatch.setattr(f"{MODULE}._make_session", lambda connection: session)
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", lambda: SecretStr("test-token"))
    batch.submit_job.return_value = {"jobId": str(uuid4())}
    batch.describe_jobs.return_value = {"jobs": [{"status": "SUCCEEDED"}]}
    s3.get_object.side_effect = lambda **kwargs: {
        "Body": BytesIO(
            ('{"status":"completed","progress":100,"updated_at":"' + datetime.now(UTC).isoformat() + '"}').encode()
        )
    }
    return batch, s3


@pytest.mark.asyncio
async def test_batch_catalog_and_preparation_do_not_use_http(connection, monkeypatch) -> None:
    trainer = RemoteTrainer(id=uuid4(), name="AWS Provider", connection=connection)
    service = RemoteTrainerService(MagicMock())
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    http = MagicMock(side_effect=AssertionError("Batch has no HTTP endpoint"))
    monkeypatch.setattr("services.remote_trainer_service.httpx.AsyncClient", http)
    monkeypatch.setattr("services.training_targets.remote._make_session", MagicMock())
    catalog = await service.get_policy_catalog(trainer.id)
    assert {p.id for p in catalog.available} == {"act", "pi05", "smolvla", "molmoact2"}
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        remote_trainer_id=trainer.id,
        policy="pi05",
        model_name="model",
        device={"type": "cuda", "index": 2},
    )
    prepared = await RemoteTrainingTargetHandler(service).prepare(payload)
    assert prepared.device is None
    assert prepared.remote_trainer_url == aws_batch_trainer_url(connection)
    assert "batch_connection" not in prepared.model_dump()
    assert RemoteTrainingTargetHandler.target_key(prepared) == f"batch:{trainer.id}:pi05"
    restored = TrainJobPayloadAdapter.validate_json(prepared.model_dump_json())
    backend = await get_training_backend(restored)
    assert isinstance(backend, AwsBatchTrainingBackend)
    assert backend._connection == connection
    http.assert_not_called()


@pytest.mark.parametrize("configuration_uri", [None, "s3://config-bucket/trainer.json"])
@pytest.mark.parametrize("profile_name", ["default", "studio"])
def test_batch_trainer_url_preserves_resolved_resources(connection, configuration_uri, profile_name) -> None:
    connection.configuration_uri = configuration_uri
    connection.profile_name = profile_name
    connection.targets["g4dn.2xlarge"].job_definition = "arn:aws:batch:us-east-1:123456789012:job-definition/small:1"
    assert aws_batch_connection_from_url(aws_batch_trainer_url(connection)) == connection


@pytest.mark.parametrize("url", ["https://trainer.test", "aws-batch://us-east-1/jobs-bucket"])
def test_invalid_batch_trainer_url_is_rejected(url) -> None:
    with pytest.raises(ValueError):
        aws_batch_connection_from_url(url)


@pytest.mark.asyncio
async def test_catalog_excludes_policies_without_a_provisioned_target(connection) -> None:
    connection.targets.pop("g7e.8xlarge")
    trainer = RemoteTrainer(id=uuid4(), name="AWS Provider", connection=connection)
    service = RemoteTrainerService(MagicMock())
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    assert "molmoact2" not in {p.id for p in (await service.get_policy_catalog(trainer.id)).available}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,queue", [("act", "small"), ("smolvla", "small"), ("pi05", "medium"), ("molmoact2", "large")]
)
async def test_submit_routes_by_policy_and_ingests_artifact(context, aws, monkeypatch, tmp_path, policy, queue) -> None:
    batch, s3 = aws
    context.payload.policy = policy
    context.model.policy = policy
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"dataset")
    monkeypatch.setattr(f"{MODULE}.DatasetDownloadService.create_dataset_archive", lambda self, path: archive)
    monkeypatch.setattr(f"{MODULE}.build_spec", lambda context: TrainingJobSpec(policy=policy))
    monkeypatch.setattr(
        f"{MODULE}.get_settings",
        lambda: MagicMock(
            data_import_max_uncompressed_bytes=1024**2,
            data_import_min_free_bytes=0,
        ),
    )

    def download(bucket, key, path, **kwargs):
        with ZipFile(path, "w") as artifact:
            artifact.writestr("model.ckpt", "trained")

    s3.download_file.side_effect = download
    await (await get_training_backend(context.payload)).train(context)
    submitted = batch.submit_job.call_args.kwargs
    assert submitted["jobQueue"] == f"arn:aws:batch:us-east-1:123456789012:job-queue/{queue}-queue"
    assert submitted["jobDefinition"] == f"arn:aws:batch:us-east-1:123456789012:job-definition/{queue}-definition"
    assert submitted["parameters"] == {"job_prefix": f"jobs/{context.job.id}"}
    assert submitted["containerOverrides"]["environment"] == [{"name": "HF_TOKEN", "value": "test-token"}]
    assert b"test-token" not in s3.put_object.call_args.kwargs["Body"]
    assert (context.output_dir / "model.ckpt").read_text() == "trained"
    context.on_remote_job_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_reattach_does_not_resubmit_or_upload(context, aws, monkeypatch) -> None:
    batch, s3 = aws
    context.remote_job_id = uuid4()
    context.snapshot = None
    backend = await get_training_backend(context.payload)
    monkeypatch.setattr(backend, "_download", MagicMock())
    monkeypatch.setattr(backend, "_extract", MagicMock())
    await backend.train(context)
    batch.submit_job.assert_not_called()
    s3.upload_file.assert_not_called()
    s3.put_object.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_jobs_use_isolated_s3_paths(context, aws, monkeypatch, tmp_path) -> None:
    batch, s3 = aws
    other = replace(context, job=MagicMock(id=uuid4(), progress=0))
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"dataset")
    monkeypatch.setattr(f"{MODULE}.DatasetDownloadService.create_dataset_archive", lambda self, path: archive)
    monkeypatch.setattr(f"{MODULE}.build_spec", lambda context: TrainingJobSpec(policy="act"))
    monkeypatch.setattr(f"{MODULE}.cleanup_staged_archive", lambda path: None)
    monkeypatch.setattr(AwsBatchTrainingBackend, "_extract", MagicMock())

    await asyncio.gather(
        (await get_training_backend(context.payload)).train(context),
        (await get_training_backend(other.payload)).train(other),
    )

    prefixes = {f"jobs/{job.job.id}" for job in (context, other)}
    assert {args.kwargs["parameters"]["job_prefix"] for args in batch.submit_job.call_args_list} == prefixes
    assert {args.args[2] for args in s3.upload_file.call_args_list} == {f"{p}/dataset.zip" for p in prefixes}
    assert {args.kwargs["Key"] for args in s3.put_object.call_args_list} == {f"{p}/spec.json" for p in prefixes}
    assert {args.kwargs["Key"] for args in s3.get_object.call_args_list} == {f"{p}/status.json" for p in prefixes}
    assert {args.args[1] for args in s3.download_file.call_args_list} == {f"{p}/artifact.zip" for p in prefixes}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel,suspend,error",
    [(True, False, TrainingCanceledError), (False, True, TrainingSuspendedError), (True, True, TrainingCanceledError)],
)
async def test_cancel_and_shutdown(context, aws, cancel, suspend, error) -> None:
    batch, _ = aws
    context.remote_job_id = uuid4()
    context.should_stop = lambda: True
    context.should_cancel_job = lambda: cancel
    context.should_suspend = lambda: suspend
    with pytest.raises(error):
        await (await get_training_backend(context.payload)).train(context)
    assert batch.terminate_job.called is cancel


@pytest.mark.asyncio
async def test_scheduler_failure_without_status_is_reported(context, aws) -> None:
    batch, s3 = aws
    context.remote_job_id = uuid4()
    batch.describe_jobs.return_value = {"jobs": [{"status": "FAILED", "statusReason": "Image pull failed"}]}
    s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    with pytest.raises(RuntimeError, match="Image pull failed"):
        await (await get_training_backend(context.payload)).train(context)


@pytest.mark.parametrize("phase", ["poll", "download", "logs"])
async def test_expired_credentials_wait_and_recover_without_resubmitting(context, aws, monkeypatch, phase) -> None:
    context.remote_job_id = uuid4()
    original_id = context.remote_job_id
    context.on_authentication_issue = AsyncMock()
    backend = AwsBatchTrainingBackend(aws_batch_connection_from_url(context.payload.remote_trainer_url))
    expired = SSOTokenLoadError(error_msg="expired")
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", AsyncMock())
    poll = MagicMock(return_value=({"status": "SUCCEEDED"}, None))
    download, logs = MagicMock(), MagicMock()
    if phase == "poll":
        poll.side_effect = [expired, ({"status": "SUCCEEDED"}, None)]
    elif phase == "download":
        download.side_effect = [expired, None]
    else:
        logs.side_effect = [expired, None]
    monkeypatch.setattr(backend, "_poll", poll)
    monkeypatch.setattr(backend, "_download", download)
    monkeypatch.setattr(backend, "_relay_logs", logs)
    monkeypatch.setattr(backend, "_extract", MagicMock())
    await backend.train(context)
    assert context.remote_job_id == original_id
    aws[0].submit_job.assert_not_called()
    issues = context.on_authentication_issue.call_args_list
    assert issues[0].args[0].command == "aws sso login --profile=default"
    assert issues[-1].args == (None,)
    assert any(
        "Awaiting AWS authentication" in call.kwargs.get("message", "") for call in context.progress.call_args_list
    )


async def test_cancel_waits_for_authentication_before_confirming_stop(context, aws, monkeypatch) -> None:
    context.remote_job_id = uuid4()
    context.should_cancel_job = lambda: True
    context.on_authentication_issue = AsyncMock()
    aws[0].terminate_job.side_effect = [SSOTokenLoadError(error_msg="expired"), None]
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", AsyncMock())
    with pytest.raises(TrainingCanceledError):
        await AwsBatchTrainingBackend(aws_batch_connection_from_url(context.payload.remote_trainer_url)).train(context)
    assert aws[0].terminate_job.call_count == 2
    assert any("not confirmed stopped" in call.kwargs.get("message", "") for call in context.progress.call_args_list)


async def test_authentication_wait_survives_shutdown(context, aws, monkeypatch) -> None:
    context.remote_job_id = uuid4()
    context.on_authentication_issue = AsyncMock()
    backend = AwsBatchTrainingBackend(aws_batch_connection_from_url(context.payload.remote_trainer_url))

    def expired():
        context.should_suspend = lambda: True
        raise SSOTokenLoadError(error_msg="expired")

    monkeypatch.setattr(backend, "_poll", lambda *args: expired())
    with pytest.raises(TrainingSuspendedError):
        await backend.train(context)
    aws[0].terminate_job.assert_not_called()


async def test_ambiguous_submission_failure_is_not_retried(context, aws, monkeypatch, tmp_path) -> None:
    from botocore.exceptions import EndpointConnectionError

    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"dataset")
    monkeypatch.setattr(f"{MODULE}.DatasetDownloadService.create_dataset_archive", lambda *args: archive)
    backend = AwsBatchTrainingBackend(aws_batch_connection_from_url(context.payload.remote_trainer_url))
    monkeypatch.setattr(backend, "_upload_inputs", MagicMock())
    submit = MagicMock(side_effect=EndpointConnectionError(endpoint_url="https://batch.amazonaws.com"))
    monkeypatch.setattr(backend, "_submit", submit)
    with pytest.raises(EndpointConnectionError):
        await backend.train(context)
    submit.assert_called_once()


async def test_submission_rejects_expired_profile_before_creating_job(connection, context, monkeypatch) -> None:
    from services.aws_auth_service import AwsAuthenticationRequiredError

    trainer = RemoteTrainer(id=uuid4(), name="AWS Provider", connection=connection)
    service = RemoteTrainerService(MagicMock())
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    monkeypatch.setattr(
        "services.training_targets.remote._make_session", MagicMock(side_effect=SSOTokenLoadError(error_msg="expired"))
    )
    with pytest.raises(AwsAuthenticationRequiredError) as captured:
        await RemoteTrainingTargetHandler(service).prepare(context.payload)
    assert captured.value.issue.command == "aws sso login --profile=default"


def test_policy_routes_match_existing_cloudformation_mapping() -> None:
    path = Path(__file__).resolve().parents[3] / "cloudformation" / "remote-trainer.yaml"
    mapping = yaml.load(path.read_text(), Loader=yaml.BaseLoader)["Mappings"]["PolicyInstanceType"]
    assert {
        label.split(" (")[0].lower().replace(".", ""): value["InstanceType"] for label, value in mapping.items()
    } == POLICY_INSTANCE_TYPES


@pytest.mark.asyncio
async def test_progress_survives_missing_status_and_retry(context, aws, monkeypatch) -> None:
    context.remote_job_id = uuid4()
    backend = await get_training_backend(context.payload)
    status = BatchJobStatus(status="running", progress=50, message="Training", updated_at=datetime.now(UTC))
    monkeypatch.setattr(
        backend,
        "_poll",
        MagicMock(
            side_effect=[
                ClientError({"Error": {"Code": "ThrottlingException"}}, "DescribeJobs"),
                ({"status": "RUNNABLE"}, None),
                ({"status": "RUNNING"}, status),
                ({"status": "SUCCEEDED"}, status),
            ]
        ),
    )
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", AsyncMock())
    monkeypatch.setattr(backend, "_download", MagicMock())
    monkeypatch.setattr(backend, "_extract", MagicMock())
    await backend.train(context)
    updates = context.progress.call_args_list
    assert [call.args[0] for call in updates] == sorted(call.args[0] for call in updates)
    assert any("AWS Batch RUNNING: Training" in call.kwargs.get("message", "") for call in updates)


@pytest.mark.asyncio
async def test_runner_failure_is_reported(context, aws) -> None:
    batch, s3 = aws
    context.remote_job_id = uuid4()
    batch.describe_jobs.return_value = {"jobs": [{"status": "FAILED", "statusReason": "Essential container exited"}]}
    status = BatchJobStatus(
        status="failed", progress=0, message="Missing policy dependency", updated_at=datetime.now(UTC)
    )
    s3.get_object.return_value = {"Body": BytesIO(status.model_dump_json().encode())}
    s3.get_object.side_effect = None
    with pytest.raises(RuntimeError, match="Missing policy dependency"):
        await (await get_training_backend(context.payload)).train(context)


@pytest.mark.asyncio
async def test_failed_job_id_persistence_terminates_submitted_job(context, aws, monkeypatch, tmp_path) -> None:
    batch, _ = aws
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"dataset")
    monkeypatch.setattr(f"{MODULE}.DatasetDownloadService.create_dataset_archive", lambda self, path: archive)
    backend = await get_training_backend(context.payload)
    monkeypatch.setattr(backend, "_upload_inputs", MagicMock())
    context.on_remote_job_id.side_effect = RuntimeError("Database unavailable")
    with pytest.raises(RuntimeError, match="Database unavailable"):
        await backend.train(context)
    batch.terminate_job.assert_called_once_with(
        jobId=batch.submit_job.return_value["jobId"], reason="Canceled from Physical AI Studio"
    )


@pytest.mark.asyncio
async def test_cancellation_interrupts_artifact_download(context, aws) -> None:
    batch, s3 = aws
    context.remote_job_id = uuid4()

    def download(bucket, key, path, **kwargs):
        context.should_stop = lambda: True
        kwargs["Callback"](1024)

    s3.download_file.side_effect = download
    with pytest.raises(TrainingCanceledError):
        await (await get_training_backend(context.payload)).train(context)
    batch.terminate_job.assert_called_once()


@pytest.mark.asyncio
async def test_reported_failure_terminates_a_container_that_has_not_exited(context, aws) -> None:
    batch, s3 = aws
    context.remote_job_id = uuid4()
    batch.describe_jobs.return_value = {"jobs": [{"status": "RUNNING"}]}
    status = BatchJobStatus(status="failed", message="CUDA OOM", updated_at=datetime.now(UTC))
    s3.get_object.return_value = {"Body": BytesIO(status.model_dump_json().encode())}
    s3.get_object.side_effect = None

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        await (await get_training_backend(context.payload)).train(context)

    batch.terminate_job.assert_called_once_with(jobId=str(context.remote_job_id), reason="Trainer reported failure")


def test_container_logs_are_relayed_incrementally(connection, monkeypatch) -> None:
    backend = AwsBatchTrainingBackend(connection)
    logs = MagicMock()
    logs.get_log_events.side_effect = [
        {"events": [{"message": "Training started"}], "nextForwardToken": "cursor-1"},
        {"events": [{"message": "CUDA OOM"}], "nextForwardToken": "cursor-2"},
    ]
    monkeypatch.setattr(backend, "_client", lambda service: logs)
    info = MagicMock()
    monkeypatch.setattr(f"{MODULE}.logger.info", info)
    job = {
        "container": {
            "logStreamName": "job-stream",
            "logConfiguration": {"options": {"awslogs-group": "trainer-logs"}},
        }
    }

    backend._relay_logs(job)
    backend._relay_logs(job)

    assert logs.get_log_events.call_args_list == [
        call(logGroupName="trainer-logs", logStreamName="job-stream", startFromHead=True),
        call(logGroupName="trainer-logs", logStreamName="job-stream", startFromHead=True, nextToken="cursor-1"),
    ]
    assert [entry.args[1] for entry in info.call_args_list] == ["Training started", "CUDA OOM"]


@pytest.mark.parametrize("code", ["ResourceNotFoundException", "AccessDeniedException"])
def test_log_read_failures_do_not_fail_training(connection, monkeypatch, code) -> None:
    backend = AwsBatchTrainingBackend(connection)
    logs = MagicMock()
    logs.get_log_events.side_effect = ClientError({"Error": {"Code": code}}, "GetLogEvents")
    monkeypatch.setattr(backend, "_client", lambda service: logs)
    backend._relay_logs(
        {"container": {"logStreamName": "stream", "logConfiguration": {"options": {"awslogs-group": "logs"}}}}
    )


@pytest.mark.asyncio
async def test_training_updates_replace_dataset_ready_and_preserve_loss(context, aws, monkeypatch) -> None:
    backend = await get_training_backend(context.payload)
    container = {"image": "ghcr.io/example/trainer:submitted-version"}
    ready = BatchJobStatus(status="running", message="Dataset ready", updated_at=datetime.now(UTC))
    training = ready.model_copy(update={"message": None, "extra_info": {"train/loss_step": 6.45}})
    monkeypatch.setattr(
        backend,
        "_poll",
        MagicMock(
            side_effect=[
                ({"status": "RUNNING", "container": container}, ready),
                ({"status": "RUNNING", "container": container}, training),
                ({"status": "RUNNING", "container": container}, training),
                ({"status": "SUCCEEDED", "container": container}, training),
            ]
        ),
    )
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", AsyncMock())

    await backend._wait(context, "batch-job-id", "jobs/studio-job-id")

    updates = context.progress.call_args_list
    assert "Dataset ready" in updates[0].kwargs["message"]
    assert "AWS Batch RUNNING: Training model" in updates[1].kwargs["message"]
    assert "AWS Batch RUNNING: Training model" in updates[2].kwargs["message"]
    assert updates[2].kwargs["extra_info"]["train/loss_step"] == 6.45
    assert updates[2].kwargs["extra_info"]["aws_batch"]["job_id"] == "batch-job-id"
    assert all(update.kwargs["extra_info"]["aws_batch"]["image"] == container["image"] for update in updates)


def test_waiting_for_compute_includes_elapsed_time_and_aws_reason(monkeypatch) -> None:
    monkeypatch.setattr(f"{MODULE}.time.time", lambda: 1000)
    message, details = AwsBatchTrainingBackend._execution_progress(
        {"status": "RUNNABLE", "createdAt": 640000, "statusReason": "CAPACITY:INSUFFICIENT_INSTANCE_CAPACITY"}, None, {}
    )
    assert "Waiting for compute capacity and scheduling" in message
    assert "6m 00s elapsed" in message
    assert "CAPACITY:INSUFFICIENT_INSTANCE_CAPACITY" in message
    assert details["elapsed_seconds"] == 360


@pytest.mark.parametrize(
    "task,expected",
    [
        ({}, "Starting container; image download and setup may take several minutes"),
        ({"pullStartedAt": datetime.now(UTC)}, "Pulling container image"),
        (
            {"pullStartedAt": datetime.now(UTC), "pullStoppedAt": datetime.now(UTC)},
            "Container image downloaded; preparing startup",
        ),
    ],
)
def test_image_pull_is_reported_only_when_ecs_confirms_it(task, expected) -> None:
    message, _ = AwsBatchTrainingBackend._execution_progress({"status": "STARTING"}, None, task)
    assert message == f"AWS Batch STARTING: {expected}"


def test_stale_trainer_report_is_distinguished_from_current_batch_state(monkeypatch) -> None:
    monkeypatch.setattr(f"{MODULE}.time.time", lambda: 1000)
    status = BatchJobStatus(status="running", message="Dataset ready", updated_at=datetime.fromtimestamp(640, UTC))
    message, details = AwsBatchTrainingBackend._execution_progress({"status": "RUNNING"}, status, {})
    assert "last trainer update 6m 00s ago" in message
    assert details["trainer_update_age_seconds"] == 360


def test_startup_diagnostics_use_the_ecs_task_cluster(connection, monkeypatch) -> None:
    backend = AwsBatchTrainingBackend(connection)
    ecs = MagicMock()
    ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "PENDING"}]}
    monkeypatch.setattr(backend, "_client", lambda service: ecs)
    arn = "arn:aws:ecs:us-east-1:123:task/batch-cluster/task-id"

    assert backend._starting_task({"container": {"taskArn": arn}}) == {"lastStatus": "PENDING"}
    ecs.describe_tasks.assert_called_once_with(cluster="batch-cluster", tasks=[arn])


def test_missing_startup_permissions_do_not_block_batch_monitoring(connection, monkeypatch) -> None:
    backend = AwsBatchTrainingBackend(connection)
    ecs = MagicMock()
    ecs.describe_tasks.side_effect = ClientError({"Error": {"Code": "AccessDeniedException"}}, "DescribeTasks")
    monkeypatch.setattr(backend, "_client", lambda service: ecs)
    assert backend._starting_task({"container": {"taskArn": "arn:aws:ecs:us-east-1:123:task/cluster/id"}}) == {}

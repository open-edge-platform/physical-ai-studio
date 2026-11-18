# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise native Modal jobs without credentials, network access, or GPUs."""

from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from uuid import uuid4
from zipfile import ZipFile

import modal
import pytest
from physicalai.data.archive_safety import ZipBombDetectedError
from pydantic import SecretStr, ValidationError

from exceptions import InvalidResourceError
from schemas.job import RemoteJobIdAdapter, RemoteTrainJobPayload, TrainingDevice, TrainJobPayloadAdapter
from schemas.remote_trainer import ModalConnection, RemoteTrainer
from services.modal_training import modal_connection_from_url, modal_trainer_url, probe_modal
from services.remote_trainer_service import RemoteTrainerService
from services.training_backends import get_training_backend
from services.training_backends.base import TrainingCanceledError, TrainingContext, TrainingSuspendedError
from services.training_backends.modal import ModalTrainingBackend, read_volume_file
from services.training_service import TrainingService
from services.training_targets.remote import RemoteTrainingTargetHandler
from trainer.batch.status import BatchJobStatus
from training import TrainingJobSpec

MODULE = "services.training_backends.modal"
CALL_ID = "fc-0123456789ABCDEF"


@pytest.fixture
def connection():
    return ModalConnection(connection_mode="modal")


@pytest.fixture
def context(tmp_path, connection):
    return TrainingContext(
        job=MagicMock(id=uuid4(), progress=0),
        model=MagicMock(policy="act"),
        snapshot=MagicMock(path=str(tmp_path / "dataset")),
        payload=RemoteTrainJobPayload(
            project_id=uuid4(),
            dataset_id=uuid4(),
            remote_trainer_id=uuid4(),
            policy="act",
            model_name="model",
            remote_trainer_url=modal_trainer_url(connection),
        ),
        base_model=None,
        output_dir=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        progress=MagicMock(),
        should_stop=lambda: False,
        on_remote_job_id=AsyncMock(),
    )


@pytest.fixture
def resources(context, monkeypatch, tmp_path):
    function, volume, call = MagicMock(), MagicMock(), MagicMock(object_id=CALL_ID)
    function.spawn.return_value = call
    function.info.return_value.secrets = []
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", MagicMock(return_value=None))
    call.get.return_value = 0
    monkeypatch.setattr(f"{MODULE}.modal_resources", lambda connection: (function, volume))
    monkeypatch.setattr("modal.FunctionCall.from_id", MagicMock(return_value=call))
    archive = BytesIO()
    with ZipFile(archive, "w") as output:
        output.writestr("model.ckpt", "trained")
    files = {
        f"jobs/{context.job.id}/status.json": BatchJobStatus(
            status="completed", progress=100, updated_at=datetime.now(UTC)
        )
        .model_dump_json()
        .encode(),
        f"jobs/{context.job.id}/artifact.zip": archive.getvalue(),
    }

    def read_file(path):
        if path not in files:
            raise FileNotFoundError(path)
        yield files[path]

    volume.read_file.side_effect = read_file
    volume.listdir.return_value = []
    source = tmp_path / "dataset.zip"
    source.write_bytes(b"dataset")
    monkeypatch.setattr(f"{MODULE}.DatasetDownloadService.create_dataset_archive", lambda self, path: source)
    monkeypatch.setattr(
        f"{MODULE}.build_spec",
        lambda context: TrainingJobSpec(policy="act", run_options={"hf_token": SecretStr("secret")}),
    )
    monkeypatch.setattr(
        f"{MODULE}.get_settings",
        lambda: MagicMock(
            data_import_max_uncompressed_bytes=1024**2,
            data_import_min_free_bytes=0,
        ),
    )
    return function, volume, call, files


@pytest.mark.asyncio
async def test_catalog_preparation_and_factory(connection, monkeypatch):
    trainer = RemoteTrainer(id=uuid4(), name="Modal", connection=connection)
    service = RemoteTrainerService(MagicMock())
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    http = MagicMock(side_effect=AssertionError("Modal must not use HTTP trainers"))
    monkeypatch.setattr("services.remote_trainer_service.httpx.AsyncClient", http)
    monkeypatch.setattr(
        "modal.Function.from_name", MagicMock(side_effect=AssertionError("Modal must not check policy capabilities"))
    )
    assert {policy.id for policy in (await service.get_policy_catalog(trainer.id)).available} == {
        "act",
        "molmoact2",
        "pi05",
        "rldx1",
        "smolvla",
        "xr0",
    }
    payload = RemoteTrainJobPayload(
        project_id=uuid4(),
        dataset_id=uuid4(),
        remote_trainer_id=trainer.id,
        policy="act",
        model_name="model",
        device={"type": "cuda", "index": 2},
    )
    prepared = await RemoteTrainingTargetHandler(service).prepare(payload)
    assert prepared.device is None
    assert modal_connection_from_url(prepared.remote_trainer_url) == connection
    assert RemoteTrainingTargetHandler.target_key(prepared) == f"batch:{trainer.id}:act"
    restored = TrainJobPayloadAdapter.validate_json(prepared.model_dump_json())
    assert isinstance(await get_training_backend(restored), ModalTrainingBackend)
    http.assert_not_called()


@pytest.mark.asyncio
async def test_gpu_override_is_rejected_for_non_modal_target(context):
    trainer = RemoteTrainer(
        id=uuid4(), name="HTTP", connection={"connection_mode": "direct", "url": "http://trainer.test"}
    )
    service = RemoteTrainerService(MagicMock())
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    service.get_policy_catalog = AsyncMock()
    context.payload.modal_gpu = "H100"
    with pytest.raises(InvalidResourceError, match="only supported on Modal"):
        await RemoteTrainingTargetHandler(service).prepare(context.payload)
    service.get_policy_catalog.assert_not_awaited()


def test_old_policy_lists_are_ignored():
    connection = ModalConnection.model_validate({"connection_mode": "modal", "policies": ["act"]})
    assert "policies" not in connection.model_dump()
    assert "policies" not in ModalConnection.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        ModalConnection.model_validate({"connection_mode": "modal", "unexpected": "value"})


def test_ids_and_resources_round_trip(connection, context):
    assert modal_connection_from_url(modal_trainer_url(connection)) == connection
    context.payload.remote_job_id = CALL_ID
    restored = TrainJobPayloadAdapter.validate_json(context.payload.model_dump_json())
    assert restored.remote_job_id == CALL_ID
    assert TrainingService._reattachable_remote_job_id(MagicMock(payload=restored)) == CALL_ID
    for invalid in ["arbitrary", "fc-../bad"]:
        with pytest.raises(ValidationError):
            RemoteJobIdAdapter.validate_python(invalid)
    for invalid in ["https://example.test", "modal://trainer", "modal://trainer?connection=invalid"]:
        with pytest.raises(ValueError):
            modal_connection_from_url(invalid)


@pytest.mark.parametrize("fails", [False, True])
def test_dataset_upload_logs_size_and_average_speed(context, resources, monkeypatch, tmp_path, fails):
    _, volume, _, _ = resources
    archive = tmp_path / "snapshot.zip"
    archive.write_bytes(b"x" * 2048)
    log = MagicMock()
    monkeypatch.setattr(f"{MODULE}.logger", log)
    monkeypatch.setattr(f"{MODULE}.time.monotonic", MagicMock(side_effect=[10.0, 14.0]))
    if fails:
        volume.batch_upload.return_value.__exit__.side_effect = RuntimeError("upload failed")
        with pytest.raises(RuntimeError, match="upload failed"):
            ModalTrainingBackend._upload(context, volume, "jobs/test", archive)
    else:
        ModalTrainingBackend._upload(context, volume, "jobs/test", archive)
    lines = [entry.args[0].format(*entry.args[1:]) for entry in log.info.call_args_list]
    assert lines[0] == "Uploading dataset snapshot to Modal Volume: snapshot.zip (2.0 KiB)"
    if fails:
        assert len(lines) == 1
    else:
        assert lines[1] == "Snapshot uploaded to Modal Volume: 2.0 KiB in 4.0s (512 B/s)"


@pytest.mark.parametrize("size_status", ["known", "missing", "unavailable"])
def test_model_download_reuses_transfer_heartbeats_and_logs_average_speed(
    context, resources, monkeypatch, tmp_path, size_status
):
    _, volume, _, _ = resources
    if size_status == "known":
        volume.listdir.return_value = [MagicMock(size=2048)]
    elif size_status == "unavailable":
        volume.listdir.side_effect = modal.exception.ConnectionError("offline")
    clock = [0.0]
    monkeypatch.setattr(f"{MODULE}.time.monotonic", lambda: clock[0])
    log = MagicMock()
    monkeypatch.setattr(f"{MODULE}.logger", log)

    def chunks(path):
        assert path == "jobs/test/artifact.zip"
        clock[0] = 20.0
        yield b"x" * 1024
        clock[0] = 40.0
        yield b"y" * 1024

    volume.read_file.side_effect = chunks
    archive = tmp_path / "artifact.zip"
    ModalTrainingBackend._download(context, volume, "jobs/test/artifact.zip", archive)
    assert archive.read_bytes() == b"x" * 1024 + b"y" * 1024
    volume.listdir.assert_called_once_with("jobs/test/artifact.zip")
    if size_status == "known":
        expected = [
            "Downloading trained model from Modal Volume: jobs/test/artifact.zip (2.0 KiB)",
            "Model download progress: 1.0 KiB/2.0 KiB (50%) at 51 B/s, ETA 20s",
            "Model download progress: 2.0 KiB/2.0 KiB (100%) at 51 B/s, ETA 0s",
        ]
    else:
        expected = [
            "Downloading trained model from Modal Volume: jobs/test/artifact.zip (unknown size)",
            "Model download progress: 1.0 KiB transferred at 51 B/s",
            "Model download progress: 2.0 KiB transferred at 51 B/s",
        ]
    expected.append("Model downloaded from Modal Volume: 2.0 KiB in 40.0s (51 B/s)")
    assert [entry.args[0].format(*entry.args[1:]) for entry in log.info.call_args_list] == expected
    assert log.warning.call_count == (1 if size_status == "unavailable" else 0)


@pytest.mark.asyncio
async def test_submission_requires_a_returned_call(context, resources, connection):
    function, _, _, _ = resources
    function.spawn.return_value = None
    with pytest.raises(RuntimeError, match="Modal did not return a training call"):
        await ModalTrainingBackend(connection).train(context)
    context.on_remote_job_id.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_device_type", ["cpu", "xpu", "cuda", None])
async def test_submit_persist_and_download(context, resources, connection, local_device_type):
    context.payload.device = TrainingDevice(type=local_device_type, index=2) if local_device_type else None
    function, volume, call, _ = resources
    await ModalTrainingBackend(connection).train(context)
    function.spawn.assert_called_once_with(str(context.job.id))
    function.with_options.assert_not_called()
    context.on_remote_job_id.assert_awaited_once_with(CALL_ID)
    assert context.remote_job_id == CALL_ID
    assert (context.output_dir / "model.ckpt").read_text() == "trained"
    uploader = volume.batch_upload.return_value.__enter__.return_value
    spec = TrainingJobSpec.model_validate_json(uploader.put_file.call_args_list[1].args[0].getvalue())
    assert spec.device_type == "cuda" and spec.device_index == 0
    assert spec.run_options.hf_token is None
    assert not (context.cache_dir.parent / "dataset.zip").exists()
    spec_json = uploader.put_file.call_args_list[1].args[0].getvalue()
    assert b"secret" not in spec_json and b"run_options" not in spec_json
    call.cancel.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu", ["A10G", "A100-80GB", "H100"])
async def test_per_job_gpu_uses_dynamic_function_and_is_persisted(context, resources, connection, gpu):
    function, _, call, _ = resources
    context.payload.modal_gpu = gpu
    overridden = function.with_options.return_value
    overridden.spawn.return_value = call
    await ModalTrainingBackend(connection).train(context)
    function.with_options.assert_called_once_with(gpu=gpu)
    function.spawn.assert_not_called()
    overridden.spawn.assert_called_once_with(str(context.job.id))
    restored = TrainJobPayloadAdapter.validate_json(context.payload.model_dump_json())
    assert restored.modal_gpu == gpu
    assert context.remote_job_id == CALL_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu", [None, "H100"])
@pytest.mark.parametrize("has_deployment_secrets", [False, True])
async def test_studio_hf_token_uses_runtime_secret(
    context, resources, connection, monkeypatch, gpu, has_deployment_secrets
):
    function, volume, call, _ = resources
    token = "hf-studio-test-token"
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", MagicMock(return_value=SecretStr(token)))
    existing = [MagicMock(object_id="st-other"), MagicMock(object_id="st-huggingface")]
    function.info.return_value.secrets = [secret.object_id for secret in existing] if has_deployment_secrets else []
    manager = MagicMock()
    manager.list.return_value = [*reversed(existing), MagicMock(object_id="st-unrelated")]
    monkeypatch.setattr(modal.Secret, "objects", manager)
    make_secret = MagicMock(wraps=modal.Secret.from_dict)
    monkeypatch.setattr(modal.Secret, "from_dict", make_secret)
    context.payload.modal_gpu = gpu
    configured = function.with_options.return_value
    target = configured.with_options.return_value if gpu else configured
    target.spawn.return_value = call

    await ModalTrainingBackend(connection).train(context)

    make_secret.assert_called_once_with({"HF_TOKEN": token})
    secrets = function.with_options.call_args.kwargs["secrets"]
    assert secrets[:-1] == (existing if has_deployment_secrets else [])
    assert token not in repr(secrets[-1])
    if has_deployment_secrets:
        manager.list.assert_called_once_with(environment_name=connection.environment_name)
    else:
        manager.list.assert_not_called()
    if gpu:
        configured.with_options.assert_called_once_with(gpu=gpu)
    else:
        configured.with_options.assert_not_called()
    target.spawn.assert_called_once_with(str(context.job.id))
    uploader = volume.batch_upload.return_value.__enter__.return_value
    assert token.encode() not in uploader.put_file.call_args_list[1].args[0].getvalue()
    assert token not in context.payload.model_dump_json()
    assert all(token not in str(event) for event in context.progress.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [None, SecretStr("")])
async def test_no_studio_token_keeps_deployed_secrets(context, resources, connection, monkeypatch, token):
    function, _, _, _ = resources
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", MagicMock(return_value=token))
    await ModalTrainingBackend(connection).train(context)
    function.info.assert_not_called()
    function.with_options.assert_not_called()
    function.spawn.assert_called_once_with(str(context.job.id))


@pytest.mark.asyncio
async def test_unnamed_deployment_secret_is_not_silently_removed(context, resources, connection, monkeypatch):
    function, volume, _, _ = resources
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", MagicMock(return_value=SecretStr("hf-studio-test-token")))
    function.info.return_value.secrets = ["st-unnamed"]
    manager = MagicMock()
    manager.list.return_value = []
    monkeypatch.setattr(modal.Secret, "objects", manager)
    with pytest.raises(ValueError, match="preserving unnamed deployment secrets"):
        await ModalTrainingBackend(connection).train(context)
    function.spawn.assert_not_called()
    volume.batch_upload.assert_not_called()


def test_invalid_gpu_is_rejected(context):
    with pytest.raises(ValidationError):
        RemoteTrainJobPayload.model_validate({**context.payload.model_dump(), "modal_gpu": "unrestricted-gpu"})


@pytest.mark.asyncio
async def test_reattach_does_not_upload_or_spawn(context, resources, connection, monkeypatch):
    function, volume, _, _ = resources
    resolver = MagicMock(side_effect=AssertionError("Reattachment must not resolve or forward new credentials"))
    monkeypatch.setattr(f"{MODULE}.resolve_hf_token", resolver)
    context.remote_job_id = CALL_ID
    context.payload.modal_gpu = "H100"
    context.snapshot = None
    await ModalTrainingBackend(connection).train(context)
    modal.FunctionCall.from_id.assert_called_once_with(CALL_ID)
    function.spawn.assert_not_called()
    function.with_options.assert_not_called()
    function.info.assert_not_called()
    resolver.assert_not_called()
    volume.batch_upload.assert_not_called()


@pytest.mark.asyncio
async def test_interrupted_download_restarts_with_unhydrated_call(context, resources, connection):
    function, volume, call, files = resources
    read_file = volume.read_file.side_effect

    def interrupted_read(path):
        if path.endswith("artifact.zip"):
            yield files[path][:10]
            context.should_stop = lambda: True
            context.should_suspend = lambda: True
        yield from read_file(path)

    volume.read_file.side_effect = interrupted_read
    with pytest.raises(TrainingSuspendedError):
        await ModalTrainingBackend(connection).train(context)
    assert context.remote_job_id == CALL_ID
    assert not context.output_dir.exists()
    call.cancel.assert_not_called()

    context.should_stop = lambda: False
    context.should_suspend = lambda: False
    context.snapshot = None
    context.job.progress = 95
    volume.read_file.side_effect = read_file
    object_id = PropertyMock(side_effect=AttributeError("Attempting to get object_id of unhydrated FunctionCall"))
    type(call).object_id = object_id

    await ModalTrainingBackend(connection).train(context)

    object_id.assert_not_called()
    modal.FunctionCall.from_id.assert_called_once_with(CALL_ID)
    function.spawn.assert_called_once()
    volume.batch_upload.assert_called_once()
    context.on_remote_job_id.assert_awaited_once_with(CALL_ID)
    call.get.assert_not_called()
    call.cancel.assert_not_called()
    assert (context.output_dir / "model.ckpt").read_text() == "trained"
    assert volume.read_file.call_args_list[-1].args == (f"jobs/{context.job.id}/artifact.zip",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suspend,cancel,expected",
    [
        (True, False, TrainingSuspendedError),
        (False, True, TrainingCanceledError),
        (True, True, TrainingCanceledError),
    ],
)
async def test_shutdown_leaves_running_but_user_cancel_wins(context, resources, connection, suspend, cancel, expected):
    function, _, call, _ = resources
    context.remote_job_id = CALL_ID
    context.should_stop = lambda: True
    context.should_suspend = lambda: suspend
    context.should_cancel_job = lambda: cancel
    with pytest.raises(expected):
        await ModalTrainingBackend(connection).train(context)
    assert call.cancel.called == cancel
    function.spawn.assert_not_called()


@pytest.mark.asyncio
async def test_persistence_failure_cancels_submitted_call(context, resources, connection):
    _, _, call, _ = resources
    context.on_remote_job_id.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        await ModalTrainingBackend(connection).train(context)
    call.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_failure_status_and_native_failure_do_not_download(context, resources, connection):
    _, _, call, files = resources
    context.remote_job_id = CALL_ID
    files.clear()
    call.get.side_effect = RuntimeError("GPU failed")
    with pytest.raises(RuntimeError, match="GPU failed"):
        await ModalTrainingBackend(connection).train(context)
    assert not context.output_dir.exists()
    files[f"jobs/{context.job.id}/status.json"] = (
        BatchJobStatus(status="failed", message="out of memory", updated_at=datetime.now(UTC))
        .model_dump_json()
        .encode()
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        await ModalTrainingBackend(connection).train(context)
    call.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_training_telemetry_is_logged_without_duplicate_polls(context, resources, connection, monkeypatch):
    _, volume, call, _ = resources
    telemetry = {"global_step": 40, "max_steps": 100, "train/loss_step": 0.5}
    updates = [
        ("Dataset ready", None),
        (None, telemetry),
        (None, telemetry),
        (None, {**telemetry, "global_step": 45, "train/loss_step": 0.4}),
        (None, {"val_event": "start", "global_step": 45, "max_steps": 100}),
        (None, {"train/loss_step": 0.3}),
        (None, {"train/loss_step": 0.3}),
        ("Model uploaded", None),
    ]
    statuses = [
        BatchJobStatus(
            status="completed" if index == len(updates) - 1 else "running",
            progress=45,
            message=message,
            extra_info=extra_info,
            updated_at=datetime.now(UTC),
        )
        .model_dump_json()
        .encode()
        for index, (message, extra_info) in enumerate(updates)
    ]
    monkeypatch.setattr(f"{MODULE}.read_volume_file", MagicMock(side_effect=statuses))
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", AsyncMock())
    log = MagicMock()
    monkeypatch.setattr(f"{MODULE}.logger", log)
    call.get.side_effect = TimeoutError

    await ModalTrainingBackend(connection)._wait(context, call, volume, f"jobs/{context.job.id}")

    assert [entry.args[1] for entry in log.info.call_args_list] == [
        "Dataset ready",
        "Training progress: step=40/100 (40%), train/loss_step=0.5",
        "Training progress: step=45/100 (45%), train/loss_step=0.4",
        "Validation started at step=45/100",
        "Remote training progress: 45%",
        "Model uploaded",
    ]
    assert [entry.kwargs["message"] for entry in context.progress.call_args_list] == [
        message or "Training model" for message, _ in updates
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [TimeoutError, modal.exception.TimeoutError])
async def test_missing_status_polls_native_call(context, resources, connection, monkeypatch, error_type):
    _, _, call, files = resources
    context.remote_job_id = CALL_ID
    del files[f"jobs/{context.job.id}/status.json"]
    call.get.side_effect = [error_type(), 0]
    sleep = AsyncMock()
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", sleep)
    await ModalTrainingBackend(connection).train(context)
    assert call.get.call_count == 2
    sleep.assert_awaited_once_with(5)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["status", "native"])
@pytest.mark.parametrize(
    "error_type",
    [modal.exception.ConnectionError, modal.exception.AuthError, modal.exception.ServiceError],
)
async def test_monitoring_recovers_without_resubmitting(context, resources, connection, monkeypatch, phase, error_type):
    function, volume, call, files = resources
    context.remote_job_id = CALL_ID
    object_id = PropertyMock(side_effect=AttributeError("Attempting to get object_id of unhydrated FunctionCall"))
    type(call).object_id = object_id
    if phase == "status":
        volume.read_file.side_effect = [
            error_type("unavailable"),
            iter([files[f"jobs/{context.job.id}/status.json"]]),
            iter([files[f"jobs/{context.job.id}/artifact.zip"]]),
        ]
    else:
        del files[f"jobs/{context.job.id}/status.json"]
        call.get.side_effect = [error_type("unavailable"), 0]
    sleep = AsyncMock()
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", sleep)
    await ModalTrainingBackend(connection).train(context)
    sleep.assert_awaited_once_with(5)
    object_id.assert_not_called()
    function.spawn.assert_not_called()
    call.cancel.assert_not_called()
    assert context.remote_job_id == CALL_ID
    assert (context.output_dir / "model.ckpt").exists()
    assert any("reconnecting" in entry.kwargs.get("message", "") for entry in context.progress.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_monitoring_outage_still_allows_cancel_or_suspend(context, resources, connection, monkeypatch, cancel):
    function, volume, call, _ = resources
    context.remote_job_id = CALL_ID
    volume.read_file.side_effect = modal.exception.ConnectionError("offline")

    async def stop_after_retry(_):
        context.should_stop = lambda: True
        context.should_cancel_job = lambda: cancel
        context.should_suspend = lambda: not cancel

    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", stop_after_retry)
    with pytest.raises(TrainingCanceledError if cancel else TrainingSuspendedError):
        await ModalTrainingBackend(connection).train(context)
    assert call.cancel.called == cancel
    function.spawn.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("suspend", [False, True])
async def test_cancellation_retries_or_remains_pending_on_shutdown(
    context, resources, connection, monkeypatch, suspend
):
    function, _, call, _ = resources
    context.remote_job_id = CALL_ID
    context.should_stop = lambda: True
    context.should_cancel_job = lambda: True
    context.should_suspend = lambda: suspend
    call.cancel.side_effect = [modal.exception.AuthError("credentials unavailable"), None]
    sleep = AsyncMock()
    monkeypatch.setattr(f"{MODULE}.asyncio.sleep", sleep)
    with pytest.raises(TrainingSuspendedError if suspend else TrainingCanceledError):
        await ModalTrainingBackend(connection).train(context)
    assert call.cancel.call_count == (1 if suspend else 2)
    assert context.remote_job_id == CALL_ID
    function.spawn.assert_not_called()
    assert "not confirmed stopped" in context.progress.call_args.kwargs["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [modal.exception.FunctionTimeoutError, modal.exception.OutputExpiredError])
async def test_terminal_native_timeouts_are_not_polled_forever(context, resources, connection, error_type):
    _, _, call, files = resources
    context.remote_job_id = CALL_ID
    del files[f"jobs/{context.job.id}/status.json"]
    call.get.side_effect = error_type("terminal")
    with pytest.raises(error_type, match="terminal"):
        await ModalTrainingBackend(connection).train(context)
    call.get.assert_called_once()
    assert not context.output_dir.exists()


@pytest.mark.asyncio
async def test_unsafe_export_is_rejected(context, resources, connection):
    _, _, _, files = resources
    context.remote_job_id = CALL_ID
    archive = BytesIO()
    with ZipFile(archive, "w") as output:
        output.writestr("../escaped.txt", "unsafe")
    files[f"jobs/{context.job.id}/artifact.zip"] = archive.getvalue()
    with pytest.raises(ZipBombDetectedError, match="unsafe entry path"):
        await ModalTrainingBackend(connection).train(context)
    assert not (context.output_dir.parent / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_metrics_deduplicate_and_wait_for_complete_rows(connection, monkeypatch):
    import json

    from services.model_metrics_service import ModelMetricsService

    function, volume = MagicMock(), MagicMock()
    volume.read_file.side_effect = [
        iter([b"epoch,train/loss\n0,1.5\n1,"]),
        iter([b"epoch,train/loss\n0,1.5\n1,0.5\n"]),
    ]
    monkeypatch.setattr("services.modal_training.modal_resources", lambda connection: (function, volume))
    monkeypatch.setattr("services.model_metrics_service.asyncio.sleep", AsyncMock())
    stream = ModelMetricsService.tail_modal_csv_file(modal_trainer_url(connection), str(uuid4()))
    try:
        assert (await anext(stream)).event == "issues"
        assert json.loads((await anext(stream)).data) == {"epoch": 0, "train_loss": 1.5}
        assert json.loads((await anext(stream)).data) == {"epoch": 1, "train_loss": 0.5}
        assert volume.read_file.call_count == 2
    finally:
        await stream.aclose()


def test_snapshot_size_limit_and_missing_file():
    volume = MagicMock()
    volume.read_file.return_value = iter([b"ab", b"cd"])
    with pytest.raises(ValueError, match="exceeds"):
        read_volume_file(volume, "status.json", max_bytes=3)
    volume.read_file.side_effect = FileNotFoundError
    assert read_volume_file(volume, "status.json") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status,reason",
    [
        (None, "healthy", None),
        (modal.exception.AuthError("credentials missing"), "degraded", "authentication_required"),
        (modal.exception.NotFoundError("missing"), "degraded", "resource_missing"),
    ],
)
async def test_health_only_hydrates_resources(connection, monkeypatch, error, status, reason):
    function, volume = MagicMock(), MagicMock()
    function.hydrate.side_effect = error
    monkeypatch.setattr("services.modal_training.modal_resources", lambda connection: (function, volume))
    result = await probe_modal(uuid4(), connection, datetime.now(UTC))
    assert result.status == status and result.reason_code == reason and result.devices == []
    function.spawn.assert_not_called()

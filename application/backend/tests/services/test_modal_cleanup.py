# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import modal
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_remote_trainer_service, get_scheduler
from api.remote_trainers import router
from schemas.base_job import JobStatus
from schemas.job import RemoteTrainJobPayload, TrainJob
from schemas.remote_trainer import ModalConnection, RemoteTrainer
from services import modal_cleanup as cleanup
from services import modal_deployment as deployment
from services.remote_trainer_service import RemoteTrainerService


@pytest.fixture
def sdk(monkeypatch):
    stop = AsyncMock()
    lookup = AsyncMock(return_value=SimpleNamespace(stop=SimpleNamespace(aio=stop)))
    calls = {call_id: AsyncMock() for call_id in ("fc-first", "fc-second")}
    from_id = MagicMock(side_effect=lambda call_id: SimpleNamespace(cancel=SimpleNamespace(aio=calls[call_id])))
    delete = AsyncMock()
    stop_deployment = AsyncMock()
    monkeypatch.setattr(modal.App, "lookup", SimpleNamespace(aio=lookup))
    monkeypatch.setattr(modal.FunctionCall, "from_id", from_id)
    monkeypatch.setattr(modal.Volume, "objects", SimpleNamespace(delete=SimpleNamespace(aio=delete)))
    monkeypatch.setattr(cleanup, "stop_modal_deployment", stop_deployment)
    return SimpleNamespace(
        stop=stop, lookup=lookup, calls=calls, from_id=from_id, delete=delete, stop_deployment=stop_deployment
    )


@pytest.mark.anyio
async def test_stops_deployment_calls_and_app_before_deleting_volume(sdk):
    connection = ModalConnection(environment_name="team")
    trainer_id = uuid4()

    async def delete(*args, **kwargs):
        sdk.stop_deployment.assert_awaited_once_with(trainer_id)
        sdk.stop.assert_awaited_once()
        for cancel in sdk.calls.values():
            cancel.assert_awaited_once()

    sdk.delete.side_effect = delete
    await cleanup.cleanup_modal_target(trainer_id, connection, ["fc-first", "fc-second", "fc-first"])
    sdk.lookup.assert_awaited_once_with(connection.app_name, environment_name="team")
    sdk.delete.assert_awaited_once_with(connection.volume_name, environment_name="team", allow_missing=True)


@pytest.mark.anyio
@pytest.mark.parametrize("step", ["deployment", "call", "lookup", "app", "volume"])
@pytest.mark.parametrize("error", [modal.exception.AuthError, modal.exception.ConnectionError, TimeoutError])
async def test_cleanup_failures_never_skip_other_attempts(sdk, step, error):
    operation = {
        "deployment": sdk.stop_deployment,
        "call": sdk.calls["fc-first"],
        "lookup": sdk.lookup,
        "app": sdk.stop,
        "volume": sdk.delete,
    }[step]
    operation.side_effect = error("unavailable")
    await cleanup.cleanup_modal_target(uuid4(), ModalConnection(), list(sdk.calls))
    sdk.stop_deployment.assert_awaited_once()
    for cancel in sdk.calls.values():
        cancel.assert_awaited_once()
    sdk.lookup.assert_awaited_once()
    sdk.delete.assert_awaited_once()
    if step != "lookup":
        sdk.stop.assert_awaited_once()


@pytest.mark.anyio
async def test_missing_app_still_attempts_volume_cleanup(sdk):
    sdk.lookup.side_effect = modal.exception.NotFoundError("already gone")
    await cleanup.cleanup_modal_target(uuid4(), ModalConnection(), [])
    sdk.stop.assert_not_awaited()
    sdk.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_hanging_app_cleanup_is_bounded_and_still_attempts_storage(sdk, monkeypatch):
    timeout = asyncio.timeout
    monkeypatch.setattr(cleanup.asyncio, "timeout", lambda _: timeout(0.01))

    async def hang():
        await asyncio.Event().wait()

    sdk.stop.side_effect = hang
    await cleanup.cleanup_modal_target(uuid4(), ModalConnection(), [])
    sdk.delete.assert_awaited_once()


@pytest.mark.anyio
async def test_stopping_local_deployment_waits_for_task_cleanup():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def deploy():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    trainer_id = uuid4()
    task = asyncio.create_task(deploy())
    deployment._deployments[trainer_id] = deployment._Deployment(task=task)
    await started.wait()
    await deployment.stop_modal_deployment(trainer_id)
    assert stopped.is_set() and task.cancelled()
    assert deployment.modal_deployment_health(trainer_id) is None


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("local_failure", [None, "job lookup", "job interruption"])
async def test_deletion_cancels_active_jobs_and_removes_target_even_when_cloud_cleanup_fails(
    monkeypatch, cleanup_fails, local_failure
):
    trainer = RemoteTrainer(id=uuid4(), name="Modal", connection=ModalConnection())
    jobs = [
        TrainJob(
            project_id=uuid4(),
            status=status,
            payload=RemoteTrainJobPayload(
                project_id=uuid4(),
                dataset_id=uuid4(),
                remote_trainer_id=trainer.id,
                policy="act",
                model_name="test",
                remote_job_id=call_id,
            ),
        )
        for status, call_id in [
            (JobStatus.PENDING, None),
            (JobStatus.RUNNING, "fc-first"),
            (JobStatus.FAILED, "fc-second"),
        ]
    ]
    signal = MagicMock()
    job_service = MagicMock(
        repo=MagicMock(get_all=AsyncMock(return_value=jobs)),
        update_remote_job_state=AsyncMock(),
        interrupt_job=AsyncMock(),
    )
    if local_failure == "job lookup":
        job_service.repo.get_all.side_effect = RuntimeError("could not read jobs")
    elif local_failure == "job interruption":
        job_service.interrupt_job.side_effect = RuntimeError("could not interrupt job")
    monkeypatch.setattr("services.job_service.JobService", MagicMock(return_value=job_service))
    session = MagicMock(rollback=AsyncMock())
    service = RemoteTrainerService(session)
    service.repo = MagicMock(get_by_id=AsyncMock(return_value=trainer), delete_by_id=AsyncMock())

    async def cleanup_cloud(*args):
        service.repo.delete_by_id.assert_awaited_once_with(trainer.id)
        if cleanup_fails:
            raise modal.exception.AuthError("expired")

    cleanup_mock = AsyncMock(side_effect=cleanup_cloud)
    monkeypatch.setattr("services.remote_trainer_service.cleanup_modal_target", cleanup_mock)
    monkeypatch.setattr(
        "services.remote_trainer_service.modal_deployment_health",
        MagicMock(return_value=SimpleNamespace(status="starting")),
    )
    await service.delete_remote_trainer(trainer.id, signal_cancel=signal)
    expected_call_ids = [] if local_failure == "job lookup" else ["fc-first", "fc-second"]
    cleanup_mock.assert_awaited_once_with(trainer.id, trainer.connection, expected_call_ids)
    if local_failure != "job lookup":
        for job in jobs[:2]:
            job_service.update_remote_job_state.assert_any_await(job.id, cancel_requested=True)
            job_service.interrupt_job.assert_any_await(job.id, signal)
        assert job_service.interrupt_job.await_count == 2
    else:
        job_service.interrupt_job.assert_not_awaited()
    filters = job_service.repo.get_all.call_args.kwargs["expressions"]
    assert str(trainer.id) in str(filters[-1].compile(compile_kwargs={"literal_binds": True}))


def test_delete_endpoint_signals_the_scheduler():
    app = FastAPI()
    app.include_router(router)
    service = SimpleNamespace(delete_remote_trainer=AsyncMock())
    scheduler = SimpleNamespace(cancel_job=MagicMock())
    app.dependency_overrides[get_remote_trainer_service] = lambda: service
    app.dependency_overrides[get_scheduler] = lambda: scheduler
    trainer_id = uuid4()
    assert TestClient(app).delete(f"/api/remote-trainers/{trainer_id}").status_code == 204
    service.delete_remote_trainer.assert_awaited_once_with(
        trainer_id, forget_locally=False, signal_cancel=scheduler.cancel_job
    )

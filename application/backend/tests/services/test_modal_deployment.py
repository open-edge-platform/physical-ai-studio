# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import modal
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_remote_trainer_service
from api.remote_trainers import router
from exceptions import ResourceInUseError
from schemas.remote_trainer import ModalDeploymentRequest, RemoteTrainer, RemoteTrainerUpdate
from services import modal_deployment as deployment
from services.remote_trainer_service import RemoteTrainerService


@pytest.fixture
def sdk(monkeypatch):
    lookup = AsyncMock(side_effect=modal.exception.NotFoundError("missing"))
    volume = SimpleNamespace(
        hydrate=SimpleNamespace(aio=AsyncMock(side_effect=modal.exception.NotFoundError("missing")))
    )
    function = SimpleNamespace(hydrate=SimpleNamespace(aio=AsyncMock()))
    secret = SimpleNamespace(hydrate=SimpleNamespace(aio=AsyncMock()))
    monkeypatch.setattr(modal.App, "lookup", SimpleNamespace(aio=lookup))
    monkeypatch.setattr(modal.Volume, "from_name", MagicMock(return_value=volume))
    monkeypatch.setattr(modal.Function, "from_name", MagicMock(return_value=function))
    monkeypatch.setattr(modal.Secret, "from_name", MagicMock(return_value=secret))
    process = SimpleNamespace(returncode=0, wait=AsyncMock(return_value=0), terminate=MagicMock(), kill=MagicMock())
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(deployment.asyncio, "create_subprocess_exec", spawn)
    return SimpleNamespace(lookup=lookup, volume=volume, secret=secret, function=function, process=process, spawn=spawn)


@pytest.fixture(autouse=True)
def clear_deployments():
    yield
    for trainer_id in list(deployment._deployments):
        deployment.forget_modal_deployment(trainer_id)


def start(**kwargs):
    request = ModalDeploymentRequest(name="Modal", confirm_deployment=True, **kwargs)
    trainer_id = uuid4()
    connection = deployment.new_modal_connection(request.environment_name)
    deployment.start_modal_deployment(trainer_id, connection, request)
    return trainer_id, connection, deployment._deployments[trainer_id].task


@pytest.mark.anyio
async def test_deploys_bundled_nvidia_worker_with_isolated_environment(sdk, monkeypatch):
    monkeypatch.setenv("PHYSICALAI_MODAL_GPU", "untrusted-inherited-value")
    previous = os.environ.copy()
    trainer_id, connection, task = start(environment_name="team", hf_secret_name="huggingface")
    health = deployment.modal_deployment_health(trainer_id)
    assert health.status == "starting"
    await task
    assert deployment.modal_deployment_health(trainer_id) is None
    args, kwargs = sdk.spawn.call_args
    assert args[1:6] == ("-m", "modal", "deploy", "--env", "team")
    assert Path(args[-1]).is_file() and args[-1].endswith("trainer/modal_app.py")
    environment = kwargs["env"]
    assert environment["PHYSICALAI_MODAL_APP"] == connection.app_name
    assert environment["PHYSICALAI_MODAL_VOLUME"] == connection.volume_name
    assert "PHYSICALAI_MODAL_GPU" not in environment
    assert environment["PHYSICALAI_MODAL_SECRET"] == "huggingface"
    assert "trainer-cuda" in environment["PHYSICALAI_MODAL_IMAGE"]
    assert os.environ == previous
    sdk.secret.hydrate.aio.assert_awaited_once()
    sdk.function.hydrate.aio.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("existing", ["app", "volume"])
async def test_existing_resources_are_never_overwritten(sdk, existing):
    if existing == "app":
        sdk.lookup.side_effect = None
    else:
        sdk.volume.hydrate.aio.side_effect = None
    trainer_id, _, task = start()
    await task
    health = deployment.modal_deployment_health(trainer_id)
    assert health.status == "degraded" and "already exists" in health.reason_code
    sdk.spawn.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("error", [modal.exception.AuthError, modal.exception.PermissionDeniedError])
async def test_auth_failure_is_sanitized_and_does_not_deploy(sdk, error):
    sdk.lookup.side_effect = error("private credential information")
    trainer_id, _, task = start()
    await task
    health = deployment.modal_deployment_health(trainer_id)
    assert health.status == "degraded" and "authentication failed" in health.reason_code
    assert "private credential" not in health.model_dump_json()
    sdk.spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_missing_secret_does_not_start_cloud_build(sdk):
    sdk.secret.hydrate.aio.side_effect = modal.exception.NotFoundError("private secret")
    trainer_id, _, task = start(hf_secret_name="missing")
    await task
    assert "secret was not found" in deployment.modal_deployment_health(trainer_id).reason_code
    sdk.spawn.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_cli_retains_actionable_health_and_never_looks_up_function(sdk):
    sdk.process.wait.return_value = 1
    trainer_id, _, task = start()
    await task
    health = deployment.modal_deployment_health(trainer_id)
    assert health.status == "degraded" and "Modal dashboard" in health.reason_code
    sdk.function.hydrate.aio.assert_not_awaited()


@pytest.mark.anyio
async def test_shutdown_terminates_local_cli_and_keeps_interrupted_status(sdk):
    waiting = asyncio.Event()
    calls = 0

    async def wait():
        nonlocal calls
        calls += 1
        if calls == 1:
            waiting.set()
            await asyncio.Event().wait()
        return -15

    sdk.process.returncode = None
    sdk.process.wait.side_effect = wait
    trainer_id, _, task = start()
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sdk.process.terminate.assert_called_once()
    assert "interrupted" in deployment.modal_deployment_health(trainer_id).reason_code


@pytest.mark.anyio
async def test_timed_out_cli_is_killed_and_reported_as_failed(sdk):
    sdk.process.returncode = None
    sdk.process.wait.side_effect = [TimeoutError(), TimeoutError(), -9]
    trainer_id, _, task = start()
    await task
    sdk.process.terminate.assert_called_once()
    sdk.process.kill.assert_called_once()
    assert "timed out" in deployment.modal_deployment_health(trainer_id).reason_code


@pytest.mark.anyio
async def test_active_deployment_cannot_be_edited():
    trainer_id = uuid4()
    deployment._deployments[trainer_id] = deployment._Deployment()
    trainer = RemoteTrainer(id=trainer_id, name="Modal", connection=deployment.new_modal_connection("main"))
    service = RemoteTrainerService(MagicMock())
    service.repo = MagicMock(get_by_id=AsyncMock(return_value=trainer))
    with pytest.raises(ResourceInUseError):
        await service.update_remote_trainer(trainer_id, RemoteTrainerUpdate(name="Changed"))
    service.repo.update.assert_not_called()
    service.repo.delete_by_id.assert_not_called()


@pytest.mark.anyio
async def test_health_uses_deployment_progress_without_probing_gpu(sdk):
    trainer_id, connection, task = start()
    trainer = RemoteTrainer(id=trainer_id, name="Modal", connection=connection)
    result = await RemoteTrainerService._probe_remote_trainer(trainer_id, trainer)
    assert result.status == "starting"
    sdk.function.hydrate.aio.assert_not_awaited()
    await task


@pytest.mark.parametrize(
    "changes",
    [
        {"confirm_deployment": False},
        {"confirm_deployment": None},
        {"gpu": "H100"},
        {"app_name": "existing"},
        {"environment_name": "../other"},
    ],
)
def test_api_rejects_unconfirmed_or_unsafe_deployment(monkeypatch, changes):
    app = FastAPI()
    app.include_router(router)
    service = MagicMock(create_remote_trainer=AsyncMock())
    app.dependency_overrides[get_remote_trainer_service] = lambda: service
    response = TestClient(app).post(
        "/api/remote-trainers/providers/modal/deploy", json={"name": "Modal", "confirm_deployment": True, **changes}
    )
    assert response.status_code == 422
    service.create_remote_trainer.assert_not_awaited()


def test_api_saves_unique_connection_then_starts_background_deployment(monkeypatch):
    app = FastAPI()
    app.include_router(router)

    async def create(config):
        return RemoteTrainer(id=uuid4(), **config.model_dump())

    service = MagicMock(create_remote_trainer=AsyncMock(side_effect=create))
    app.dependency_overrides[get_remote_trainer_service] = lambda: service
    launch = MagicMock()
    monkeypatch.setattr("api.remote_trainers.start_modal_deployment", launch)
    client = TestClient(app)
    payload = {"name": "New Modal", "confirm_deployment": True}
    response = client.post("/api/remote-trainers/providers/modal/deploy", json=payload)
    second = client.post("/api/remote-trainers/providers/modal/deploy", json=payload)
    assert response.status_code == 202
    connection = response.json()["connection"]
    assert connection["connection_mode"] == "modal" and connection["function_name"] == "train"
    assert second.json()["connection"]["app_name"] != connection["app_name"]
    assert launch.call_count == 2

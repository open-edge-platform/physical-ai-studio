"""Policy discovery uses the selected trainer's environment, not Studio's."""

from importlib import metadata
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from exceptions import InvalidResourceError
from schemas.job import RemoteTrainJobPayload
from schemas.remote_trainer import RemoteTrainer, RemoteTrainerConnectionMode
from services.remote_trainer_service import RemoteTrainerService
from services.training_targets.remote import RemoteTrainingTargetHandler
from trainer.api import submit_job
from trainer.main import app as trainer_app
from trainer.queue_worker import QueueManager
from trainer.schemas import SubmitJobRequest
from training import TrainingJobSpec
from training.policy_catalog import AvailablePolicy, PolicyCatalog


@pytest.mark.asyncio
async def test_remote_catalog_and_submission_do_not_depend_on_local_policy_extras(monkeypatch) -> None:
    trainer_id = uuid4()
    service = RemoteTrainerService.__new__(RemoteTrainerService)
    service.get_remote_trainer = AsyncMock(
        return_value=RemoteTrainer(id=trainer_id, name="gpu", url="https://trainer.test")
    )
    catalog = PolicyCatalog(
        available=[AvailablePolicy(id="pi05", name="Pi0.5", description="remote", min_vram=None)],
        backends={"pi05": ["torch"]},
    )
    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return original_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=catalog.model_dump())),
        )

    monkeypatch.setattr("services.remote_trainer_service.httpx.AsyncClient", client)
    assert (await service.get_policy_catalog(trainer_id)).available[0].id == "pi05"


@pytest.mark.asyncio
async def test_unreachable_remote_catalog_fails_closed(monkeypatch) -> None:
    trainer_id = uuid4()
    service = RemoteTrainerService.__new__(RemoteTrainerService)
    service.get_remote_trainer = AsyncMock(
        return_value=RemoteTrainer(id=trainer_id, name="gpu", url="https://trainer.test")
    )
    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return original_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        )

    monkeypatch.setattr("services.remote_trainer_service.httpx.AsyncClient", client)
    with pytest.raises(InvalidResourceError, match="Cannot verify"):
        await service.get_policy_catalog(trainer_id)


@pytest.mark.asyncio
async def test_remote_submission_rechecks_policy_before_saving() -> None:
    trainer_id = uuid4()
    service = AsyncMock()
    service.get_remote_trainer.return_value = RemoteTrainer(id=trainer_id, name="gpu", url="https://trainer.test")
    service.get_policy_catalog.return_value = PolicyCatalog(available=[], backends={})
    payload = RemoteTrainJobPayload(
        remote_trainer_id=trainer_id, project_id=uuid4(), dataset_id=uuid4(), policy="pi05", model_name="model"
    )
    with pytest.raises(InvalidResourceError, match="not available"):
        await RemoteTrainingTargetHandler(service).prepare(payload)


@pytest.mark.asyncio
async def test_refresh_does_not_replace_image_while_studio_has_active_jobs() -> None:
    trainer_id = uuid4()
    trainer = RemoteTrainer(
        id=trainer_id,
        name="gpu",
        url="http://127.0.0.1:8001",
        connection_mode=RemoteTrainerConnectionMode.SSH,
        ssh_host_alias="gpu-box",
        ssh_local_port=8001,
        ssh_remote_port=8001,
    )
    service = RemoteTrainerService.__new__(RemoteTrainerService)
    service.get_remote_trainer = AsyncMock(return_value=trainer)
    service._require_no_active_jobs = AsyncMock(side_effect=InvalidResourceError("trainer", "active jobs"))
    with (
        patch.object(service, "_require_ssh_feature_if_tunneled"),
        patch("services.remote_trainer_service.persistent_trainer.start", new_callable=AsyncMock) as start,
        pytest.raises(InvalidResourceError, match="active jobs"),
    ):
        await service.refresh_trainer_image(trainer_id)
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_rejects_active_jobs_and_blocks_submissions() -> None:
    manager = QueueManager.__new__(QueueManager)
    manager.store = MagicMock()
    manager.store.has_active_jobs.side_effect = [True, False]
    manager.draining = False
    assert manager.begin_drain() is False
    assert manager.draining is False
    assert manager.begin_drain() is True
    request = MagicMock()
    request.app.state.queue_manager = manager
    with pytest.raises(HTTPException) as exc:
        await submit_job(SubmitJobRequest(spec=TrainingJobSpec(policy="act")), request)
    assert exc.value.status_code == 409
    manager.store.create.assert_not_called()


def test_drain_is_not_exposed_on_unmanaged_trainers(monkeypatch) -> None:
    monkeypatch.delenv("TRAINER_MANAGED", raising=False)
    client = TestClient(trainer_app)
    assert client.post("/maintenance/drain").status_code == 404
    assert client.post("/maintenance/resume").status_code == 404


def test_trainer_reports_its_own_policy_extras(monkeypatch) -> None:
    installed_version = metadata.version

    def version(name: str) -> str:
        if name == "sentencepiece":
            raise metadata.PackageNotFoundError(name)
        return installed_version(name)

    monkeypatch.setattr("training.policy_catalog.metadata.version", version)
    response = TestClient(trainer_app).get("/policies")
    assert response.status_code == 200
    ids = {policy["id"] for policy in response.json()["available"]}
    assert "act" in ids
    assert "pi05" not in ids


def test_trainer_rejects_missing_policy_extra(monkeypatch) -> None:
    monkeypatch.setattr("training.policy_catalog.missing_policy_dependencies", lambda name: ["sentencepiece"])
    response = TestClient(trainer_app).post("/jobs", json={"spec": {"policy": "pi05"}})
    assert response.status_code == 422
    assert "sentencepiece" in response.json()["detail"]

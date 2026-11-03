"""Policy discovery uses the selected trainer's environment, not Studio's."""

from importlib import metadata
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from exceptions import InvalidResourceError
from schemas.job import RemoteTrainJobPayload
from schemas.remote_trainer import DirectConnection, RemoteTrainer
from services.remote_trainer_service import RemoteTrainerService
from services.training_targets.remote import RemoteTrainingTargetHandler
from trainer.main import app as trainer_app
from training.policy_catalog import AvailablePolicy, PolicyCatalog


@pytest.mark.asyncio
async def test_remote_catalog_and_submission_do_not_depend_on_local_policy_extras(monkeypatch) -> None:
    trainer_id = uuid4()
    service = RemoteTrainerService.__new__(RemoteTrainerService)
    service.get_remote_trainer = AsyncMock(
        return_value=RemoteTrainer(id=trainer_id, name="gpu", connection=DirectConnection(url="https://trainer.test"))
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
        return_value=RemoteTrainer(id=trainer_id, name="gpu", connection=DirectConnection(url="https://trainer.test"))
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
    service.get_remote_trainer.return_value = RemoteTrainer(
        id=trainer_id, name="gpu", connection=DirectConnection(url="https://trainer.test")
    )
    service.get_policy_catalog.return_value = PolicyCatalog(available=[], backends={})
    payload = RemoteTrainJobPayload(
        remote_trainer_id=trainer_id, project_id=uuid4(), dataset_id=uuid4(), policy="pi05", model_name="model"
    )
    with pytest.raises(InvalidResourceError, match="not available"):
        await RemoteTrainingTargetHandler(service).prepare(payload)


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
    with patch("training.policy_catalog.available_policy_classes", return_value={}):
        response = TestClient(trainer_app).post("/jobs", json={"spec": {"policy": "pi05"}})
    assert response.status_code == 422
    assert "sentencepiece" in response.json()["detail"]


def test_trainer_rejects_policy_that_cannot_be_imported() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["act"]),
        patch("physicalai.policies.get_physicalai_policy_class", side_effect=ImportError("missing dependency")),
    ):
        client = TestClient(trainer_app)
        assert client.get("/policies").json()["available"] == []
        response = client.post("/jobs", json={"spec": {"policy": "act"}})
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_trainer_rejects_unknown_policy_before_queueing() -> None:
    response = TestClient(trainer_app).post("/jobs", json={"spec": {"policy": "unknown"}})
    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]


def test_trainer_rejects_discovered_policy_not_supported_by_studio() -> None:
    with (
        patch("physicalai.policies.list_physicalai_policies", return_value=["cosmos3"]),
        patch("physicalai.policies.get_physicalai_policy_class") as load_policy,
    ):
        client = TestClient(trainer_app)
        assert client.get("/policies").json()["available"] == []
        response = client.post("/jobs", json={"spec": {"policy": "cosmos3"}})

    assert response.status_code == 422
    assert "unavailable" in response.json()["detail"]
    load_policy.assert_not_called()

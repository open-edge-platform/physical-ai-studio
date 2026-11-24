from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.dependencies import get_job_service, get_model_service
from api.job import router as job_router
from api.models import router as model_router
from db.schema import Base, JobDB, ProjectDB
from repositories.mappers.model_mapper import ModelMapper
from schemas.base_job import JobStatus, JobType
from schemas.job import LocalTrainJobPayload, RemoteTrainJobPayload, TrainJob
from schemas.job_response import job_response
from schemas.model import Model
from services.job_service import JobService
from services.model_service import ModelService


@pytest.mark.anyio
@pytest.mark.parametrize("remote", [False, True])
async def test_missing_policy_history_survives_reads_and_status_updates(remote: bool) -> None:
    engine = create_async_engine("sqlite+aiosqlite://")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    project_id, job_id = uuid4(), uuid4()
    payload = {
        "project_id": str(project_id),
        "dataset_id": str(uuid4()),
        "policy": "branch_only_policy",
        "model_name": "Branch model",
        "training_target": "remote" if remote else "local",
        "lora_enabled": True,
        "snapflow_enabled": True,
        "max_epochs": 8,
        "max_steps": 321,
        "image_key_reorder_map": {"observation.images.front": 0},
        "future_plugin_setting": {"mode": "experimental", "layers": [1, 3]},
    }
    if remote:
        payload["remote_trainer_id"] = str(uuid4())
        payload["remote_job_id"] = str(uuid4())
    try:
        async with factory() as session:
            session.add(ProjectDB(id=str(project_id), name="Branch tests"))
            await session.commit()
            session.add(
                JobDB(
                    id=str(job_id),
                    project_id=str(project_id),
                    type=JobType.TRAINING,
                    status=JobStatus.FAILED,
                    progress=0,
                    message="Branch run",
                    payload=payload,
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()
            service = JobService(session)
            jobs = await service.get_job_list()
            job = jobs[0]
            assert isinstance(job, TrainJob)
            assert isinstance(job.payload, RemoteTrainJobPayload if remote else LocalTrainJobPayload)
            snapshot = job_response(job).model_dump(mode="json")
            assert snapshot["payload"]["future_plugin_setting"] == payload["future_plugin_setting"]
            assert snapshot["payload"]["max_steps"] == 321
            assert snapshot["payload"]["image_key_reorder_map"] == payload["image_key_reorder_map"]
            assert snapshot["payload"]["policy"] == "branch_only_policy"
            assert "policy_reference" not in snapshot
            updated = await service.update_job_status(job_id, JobStatus.FAILED, message="History retained")
            assert updated.payload.model_dump()["future_plugin_setting"] == payload["future_plugin_setting"]

            model = Model(
                name="Branch model",
                path="/nonexistent/branch-model",
                policy="branch_only_policy",
                properties={},
                project_id=project_id,
                train_job_id=job_id,
            )
            model_service = MagicMock()
            model_service.get_model_by_id = AsyncMock(return_value=model)
            model_service.get_backend_details.return_value = []
            model_service.get_hparams.return_value = None
            model_service.get_training_summary.side_effect = ModelService.get_training_summary
            app = FastAPI()
            app.include_router(job_router)
            app.include_router(model_router)
            app.dependency_overrides[get_job_service] = lambda: service
            app.dependency_overrides[get_model_service] = lambda: model_service
            client = TestClient(app)
            assert client.get("/api/jobs").status_code == 200
            detail = client.get(f"/api/jobs/{job_id}")
            assert detail.status_code == 200
            assert detail.json()["payload"]["future_plugin_setting"] == payload["future_plugin_setting"]
            model_detail = client.get(f"/api/models/{model.id}")
            assert model_detail.status_code == 200
            assert model_detail.json()["training_summary"]["lora_enabled"] is True
            assert model_detail.json()["model"]["policy"] == "branch_only_policy"
            assert "policy_reference" not in model_detail.json()["model"]
            with patch("training.policy_catalog.is_local_policy_available") as describe:
                assert ModelMapper.to_schema(model).policy == "branch_only_policy"
                describe.assert_not_called()
            await service.delete_job(job_id)
            assert await service.get_job_list() == []
    finally:
        await engine.dispose()

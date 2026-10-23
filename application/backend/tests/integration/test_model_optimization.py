# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing as mp
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from physicalai.inference.manifest import Manifest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.job import interrupt_job
from db.engine import get_async_db_session
from db.schema import Base, JobDB, ModelDB, ProjectDB
from exceptions import DuplicateJobException, InvalidResourceError
from main import app
from repositories.job_repo import JobRepository
from schemas.base_job import JobStatus, JobType
from schemas.dataset import Dataset
from schemas.model import Model
from services.event_processor import EventType
from services.job_service import JobService
from services.model_download_service import ModelDownloadService
from services.model_import_service import ModelImportService
from services.model_optimization_service import ModelOptimizationService
from services.model_service import ModelService
from settings import get_settings
from workers.model_optimization_worker import ModelOptimizationWorker

pytestmark = pytest.mark.anyio


@pytest.fixture
async def optimization(tmp_path, monkeypatch):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session_scope():
        async with factory() as session:
            yield session

    async def dependency():
        async with session_scope() as session:
            yield session

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    settings = SimpleNamespace(models_dir=models_dir)
    monkeypatch.setattr("services.model_optimization_service.get_settings", lambda: settings)
    monkeypatch.setattr("workers.model_optimization_worker.get_settings", lambda: settings)
    monkeypatch.setattr("workers.model_optimization_worker.get_async_db_session_ctx", session_scope)
    monkeypatch.setitem(app.dependency_overrides, get_async_db_session, dependency)
    monkeypatch.setitem(app.dependency_overrides, get_settings, lambda: settings)

    source = Model(
        name="Test policy",
        path=str(models_dir / str(uuid4())),
        project_id=uuid4(),
        policy="act",
        properties={"lora_enabled": True},
    )
    export_dir = Path(source.path) / "exports" / "openvino"
    export_dir.mkdir(parents=True)
    (export_dir / "act.xml").write_text("policy graph")
    (export_dir / "act.bin").write_bytes(b"original weights")
    (export_dir / "tokenizer.xml").write_text("tokenizer graph")
    (export_dir / "tokenizer.bin").write_bytes(b"tokenizer weights")
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "policy_package",
                "policy": {"name": "act"},
                "model": {"artifacts": {"openvino": "act.xml"}},
                "studio": {"note": "preserved"},
                "other_extension": {"enabled": True},
            }
        )
    )
    async with session_scope() as session:
        session.add(ProjectDB(id=str(source.project_id), name="Test project"))
        await session.commit()
        await ModelService(session).create_model(source)

    compressed = []

    def compress(path):
        compressed.append(Path(path))
        Path(path).with_suffix(".bin").write_bytes(b"INT8")

    monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    yield SimpleNamespace(
        sessions=session_scope, source=source, models_dir=models_dir, compressed=compressed, compress=compress
    )
    await engine.dispose()


def make_worker(job):
    return ModelOptimizationWorker(job.id, mp.Event(), mp.Event(), MagicMock())


async def test_optimize_api_creates_independent_model_and_preserves_original(optimization):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/api/models/{optimization.source.id}/optimize")
        assert response.status_code == 202
        assert response.json()["type"] == "optimize_model"
        assert response.json()["payload"] == {"model_id": str(optimization.source.id), "mode": "int8_sym"}
        duplicate = await client.post(f"/api/models/{optimization.source.id}/optimize")
        assert duplicate.status_code == 409
        async with optimization.sessions() as session:
            job = await JobService(session).get_job_by_id(response.json()["id"])
        worker = make_worker(job)
        await worker.run_loop()
        detail = await client.get(f"/api/models/{job.id}")
        assert detail.status_code == 200
        optimized = detail.json()["model"]
        assert optimized["optimization"] == {"source_model_id": str(optimization.source.id), "mode": "int8_sym"}
        assert optimized["available_backends"] == ["openvino"]
        assert optimized["snapshot_id"] is None
        assert optimized["parent_model_id"] is None
        assert optimized["lora_enabled"] is True
        reoptimize = await client.post(f"/api/models/{job.id}/optimize")
        assert reoptimize.status_code == 409

    original_export = Path(optimization.source.path) / "exports" / "openvino"
    optimized_export = optimization.models_dir / str(job.id) / "exports" / "openvino"
    assert (original_export / "act.bin").read_bytes() == b"original weights"
    assert (optimized_export / "act.bin").read_bytes() == b"INT8"
    assert (optimized_export / "tokenizer.bin").read_bytes() == b"tokenizer weights"
    original_manifest = json.loads((original_export / "manifest.json").read_text())
    optimized_manifest = json.loads((optimized_export / "manifest.json").read_text())
    assert original_manifest["studio"] == {"note": "preserved"}
    assert optimized_manifest["studio"].pop("optimization") == optimized["optimization"]
    assert optimized_manifest == original_manifest
    assert optimization.compressed == [optimization.models_dir / f".optimize-{job.id}" / "exports/openvino/act.xml"]
    async with optimization.sessions() as session:
        finished = await JobService(session).get_job_by_id(job.id)
        assert finished.status == JobStatus.COMPLETED
        assert finished.progress == 100
        assert len(await ModelService(session).get_model_list()) == 2
    assert [call.args[0][0] for call in worker.queue.put.call_args_list] == [
        EventType.JOB_UPDATE,
        EventType.MODEL_UPDATE,
        EventType.JOB_UPDATE,
    ]


@pytest.mark.parametrize("exports_root", [True, False])
async def test_optimization_marker_survives_download_and_import(optimization, tmp_path, monkeypatch, exports_root):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
    await make_worker(job).run_loop()
    model_dir = optimization.models_dir / str(job.id)
    archive = ModelDownloadService().create_model_archive(model_dir)
    restored_dir = tmp_path / "downloaded"
    try:
        with zipfile.ZipFile(archive) as downloaded:
            downloaded.extractall(restored_dir)
    finally:
        archive.unlink()
    manifest = Manifest.load(restored_dir / "exports/openvino")
    metadata = {"source_model_id": str(optimization.source.id), "mode": "int8_sym"}
    assert manifest.model_extra["studio"]["optimization"] == metadata
    # Runtime can load/save the extension without discarding Studio's classification.
    manifest.save(restored_dir / "exports/openvino/manifest.json")
    dataset = Dataset(
        name="Import context",
        project_id=optimization.source.project_id,
        environment_id=uuid4(),
        default_task="test",
    )
    importer = ModelImportService(
        get_dataset=AsyncMock(return_value=dataset),
        persist_import=AsyncMock(side_effect=lambda _job, model: model),
    )
    monkeypatch.setattr(
        "services.model_import_service.get_settings", lambda: SimpleNamespace(models_dir=optimization.models_dir)
    )
    imported = await importer.import_model_directory(
        source_dir=restored_dir / "exports" if exports_root else restored_dir,
        project_id=dataset.project_id,
        dataset_id=dataset.id,
        model_name="Restored INT8",
    )
    assert imported.optimization is not None
    assert imported.optimization.model_dump(mode="json") == metadata
    assert imported.available_backends == ["openvino"]
    assert not (Path(imported.path) / "model.ckpt").exists()
    assert (Path(imported.path) / "exports/openvino/act.bin").read_bytes() == b"INT8"


async def test_invalid_studio_manifest_metadata_does_not_touch_original(optimization):
    manifest_path = Path(optimization.source.path) / "exports/openvino/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["studio"] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    original = manifest_path.read_bytes()
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
    await make_worker(job).run_loop()
    async with optimization.sessions() as session:
        failed = await JobService(session).get_job_by_id(job.id)
        assert failed.status == JobStatus.FAILED
        assert "studio metadata must be an object" in failed.message
    assert manifest_path.read_bytes() == original
    assert not optimization.compressed
    assert not (optimization.models_dir / str(job.id)).exists()
    assert not (optimization.models_dir / f".optimize-{job.id}").exists()


async def test_train_api_rejects_optimized_base_before_creating_job(optimization, monkeypatch):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
    await make_worker(job).run_loop()
    monkeypatch.setattr("services.training_targets.local.available_policy_classes", lambda: {"act": (object, [])})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/jobs:train",
            json={
                "training_target": "local",
                "project_id": str(optimization.source.project_id),
                "dataset_id": str(uuid4()),
                "policy": "act",
                "model_name": "Unsupported optimized retrain",
                "base_model_id": str(job.id),
            },
        )
    assert response.status_code == 409
    assert "inference-only" in response.text
    async with optimization.sessions() as session:
        jobs = await JobService(session).get_job_list()
        assert len(jobs) == 1
        assert jobs[0].id == job.id


@pytest.mark.parametrize("phase", ["queued", "before_compression", "during_compression", "failure"])
async def test_cancellation_and_failure_never_publish_partial_model(optimization, monkeypatch, phase):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
        worker = make_worker(job)
        if phase == "queued":
            await interrupt_job(job.id, JobService(session), MagicMock())
        elif phase == "before_compression":
            worker.cancel_event.set()
        else:

            def compress(path):
                optimization.compress(path)
                if phase == "failure":
                    raise RuntimeError("compression failed")
                worker.cancel_event.set()

            monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    await worker.run_loop()
    async with optimization.sessions() as session:
        updated = await JobService(session).get_job_by_id(job.id)
        assert updated.status == (JobStatus.FAILED if phase == "failure" else JobStatus.CANCELED)
        assert len(await ModelService(session).get_model_list()) == 1
        if phase == "failure":
            retry = await ModelOptimizationService(session).submit(optimization.source.id)
            assert retry.id != job.id
    assert not (optimization.models_dir / str(job.id)).exists()
    assert not (optimization.models_dir / f".optimize-{job.id}").exists()
    assert (Path(optimization.source.path) / "exports/openvino/act.bin").read_bytes() == b"original weights"


@pytest.mark.parametrize("artifact", ["missing.xml", "../outside.xml", "/tmp/outside.xml", "tokenizer.bin", None])
async def test_rejects_missing_and_unsafe_artifacts(optimization, artifact):
    manifest = Path(optimization.source.path) / "exports/openvino/manifest.json"
    manifest.write_text(json.dumps({"model": {"artifacts": {"openvino": artifact}}}))
    async with optimization.sessions() as session:
        with pytest.raises(InvalidResourceError, match="complete OpenVINO export"):
            await ModelOptimizationService(session).submit(optimization.source.id)
        assert await JobService(session).get_job_list() == []


async def test_failed_publication_rolls_back_model_and_cleans_output(optimization, monkeypatch):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
    merge = AsyncSession.merge

    async def fail_completion(self, instance, **kwargs):
        if isinstance(instance, JobDB) and instance.status == JobStatus.COMPLETED:
            raise RuntimeError("Database write failed")
        return await merge(self, instance, **kwargs)

    monkeypatch.setattr(AsyncSession, "merge", fail_completion)
    await make_worker(job).run_loop()
    async with optimization.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.FAILED
        assert len(await ModelService(session).get_model_list()) == 1
    assert not (optimization.models_dir / str(job.id)).exists()
    assert not (optimization.models_dir / f".optimize-{job.id}").exists()


@pytest.mark.parametrize("failure", ["refresh", "session_close"])
async def test_post_commit_failure_preserves_published_model(optimization, monkeypatch, failure):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
    commit = AsyncSession.commit
    get_by_id = JobRepository.get_by_id
    close = AsyncSession.close

    async def track_publication(self):
        publishing = await self.scalar(select(ModelDB.id).where(ModelDB.id == str(job.id))) is not None
        await commit(self)
        self.info["published"] = publishing

    async def fail_post_commit_read(self, obj_id):
        if self.db.info.get("published"):
            raise RuntimeError("Database unavailable after commit")
        return await get_by_id(self, obj_id)

    async def fail_session_close(self):
        await close(self)
        if self.info.get("published"):
            raise RuntimeError("Database unavailable while closing published session")

    with monkeypatch.context() as patch:
        patch.setattr(AsyncSession, "commit", track_publication)
        if failure == "refresh":
            patch.setattr(JobRepository, "get_by_id", fail_post_commit_read)
        else:
            patch.setattr(AsyncSession, "close", fail_session_close)
        await make_worker(job).run_loop()

    async with optimization.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.COMPLETED
        model = await ModelService(session).get_model_by_id(job.id)
        assert model.available_backends == ["openvino"]
    assert (optimization.models_dir / str(job.id) / "exports/openvino/act.bin").read_bytes() == b"INT8"


async def test_worker_claim_is_atomic_against_queued_cancellation(optimization):
    async with optimization.sessions() as session:
        job = await ModelOptimizationService(session).submit(optimization.source.id)
        service = JobService(session)
        assert await service.claim_pending_job(job.id, JobType.TRAINING) is None
        assert await service.claim_pending_job(job.id, JobType.OPTIMIZE_MODEL) is not None
        assert await service.claim_pending_job(job.id, JobType.OPTIMIZE_MODEL) is None
        assert await service.cancel_pending_job(job.id, JobType.OPTIMIZE_MODEL) is None
        with pytest.raises(DuplicateJobException):
            await ModelOptimizationService(session).submit(optimization.source.id)

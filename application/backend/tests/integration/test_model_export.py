# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing as mp
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql.dml import Update

from db.engine import get_async_db_session
from db.schema import Base, ProjectDB
from exceptions import DuplicateJobException
from main import app
from schemas.base_job import JobStatus
from schemas.model import Model, ModelExportAction
from schemas.model_export import ExportPackage
from services.job_handlers import get_job_handler
from services.job_service import JobService
from services.model_export_service import ModelExportService
from services.model_service import ModelService
from workers.model_export_worker import ModelExportWorker

pytestmark = pytest.mark.anyio


@pytest.fixture
async def exports(tmp_path, monkeypatch):
    from physicalai.export import ExportablePolicyMixin

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def sessions():
        async with factory() as session:
            yield session

    async def dependency():
        async with sessions() as session:
            yield session

    models_dir = tmp_path / "models"
    root = models_dir / str(uuid4())
    original = root / "exports/openvino"
    original.mkdir(parents=True)
    (root / "model.ckpt").write_bytes(b"trusted checkpoint")
    (original / "model.xml").write_text("graph")
    (original / "model.bin").write_bytes(b"original weights")
    (original / "manifest.json").write_text(json.dumps({"model": {"artifacts": {"openvino": "model.xml"}}}))
    source = Model(name="Test", path=str(root), policy="act", project_id=uuid4(), properties={})
    loaded = []

    class FakePolicy(ExportablePolicyMixin):
        @staticmethod
        def get_supported_export_backends():
            return ["torch", "onnx"]

        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            loaded.append((path, kwargs))
            return cls()

        def eval(self):
            return self

        def export(self, directory, backend):
            artifact = "model.pt" if backend == "torch" else "model.onnx"
            (directory / artifact).write_bytes(b"exported weights")
            (directory / "manifest.json").write_text(json.dumps({"model": {"artifacts": {backend: artifact}}}))

    monkeypatch.setattr("physicalai.policies.get_physicalai_policy_class", lambda _name: FakePolicy)
    monkeypatch.setattr("services.model_export_service.get_settings", lambda: SimpleNamespace(models_dir=models_dir))
    monkeypatch.setattr("workers.model_export_worker.get_async_db_session_ctx", sessions)
    monkeypatch.setitem(app.dependency_overrides, get_async_db_session, dependency)
    async with sessions() as session:
        session.add(ProjectDB(id=str(source.project_id), name="Test"))
        await session.commit()
        await ModelService(session).create_model(source)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield SimpleNamespace(
            sessions=sessions,
            source=source,
            root=root,
            original=original,
            policy=FakePolicy,
            loaded=loaded,
            client=client,
            target=root / "exports/torch",
        )
    await engine.dispose()


async def submit(context):
    async with context.sessions() as session:
        return await ModelExportService(session).submit(context.source.id, ModelExportAction(backend="torch"))


def worker(job):
    return ModelExportWorker(job.id, mp.Event(), mp.Event(), MagicMock())


async def test_missing_backend_export_preserves_original_and_model_identity(exports):
    before = {path.name: path.read_bytes() for path in exports.original.iterdir()}
    detail = (await exports.client.get(f"/api/models/{exports.source.id}")).json()
    assert {item["backend"] for item in detail["available_actions"]} == {"torch", "onnx"}
    response = await exports.client.post(f"/api/models/{exports.source.id}/exports/torch")
    assert response.status_code == 202
    assert response.json()["type"] == "model_export"
    async with exports.sessions() as session:
        job = await JobService(session).get_job_by_id(response.json()["id"])
    await worker(job).run_loop()
    assert exports.loaded == [(str(exports.root / "model.ckpt"), {"map_location": "cpu", "compile_model": False})]
    assert {path.name: path.read_bytes() for path in exports.original.iterdir()} == before
    assert ExportPackage(exports.target, "torch").artifact.read_bytes() == b"exported weights"
    assert ExportPackage(exports.target, "torch").info.job_id == job.id
    assert not ModelExportService.staging(job.id).exists()
    detail = (await exports.client.get(f"/api/models/{exports.source.id}")).json()
    assert detail["model"]["id"] == str(exports.source.id)
    assert {item["type"] for item in detail["exports"]} == {"torch", "openvino"}
    assert {item["backend"] for item in detail["available_actions"]} == {"onnx"}
    assert (await exports.client.post(f"/api/models/{exports.source.id}/exports/torch")).status_code == 409


@pytest.mark.parametrize("reason", ["checkpoint", "dependencies", "backend", "unsupported", "existing"])
async def test_unavailable_exports_are_not_offered_or_queued(exports, monkeypatch, reason):
    backend = "torch"
    if reason == "checkpoint":
        (exports.root / "model.ckpt").unlink()
    elif reason == "dependencies":
        monkeypatch.setattr("training.policy_catalog.missing_policy_dependencies", lambda _name: ["missing"])
    elif reason == "backend":
        monkeypatch.setattr("services.model_service.find_spec", lambda _name: None)
    elif reason == "unsupported":
        backend = "executorch"
    else:
        backend = "openvino"
    assert backend not in {action.backend for action in ModelService.get_export_actions(exports.source)}
    assert (await exports.client.post(f"/api/models/{exports.source.id}/exports/{backend}")).status_code == 409


async def test_concurrent_submissions_reserve_only_one_export(exports):
    results = await asyncio.gather(
        *(exports.client.post(f"/api/models/{exports.source.id}/exports/torch") for _ in range(3))
    )
    assert sorted(result.status_code for result in results) == [202, 409, 409]


@pytest.mark.parametrize("phase", ["queued", "before_export", "during_export", "failure"])
async def test_cancellation_and_failure_never_publish_and_can_retry(exports, monkeypatch, phase):
    job = await submit(exports)
    process = worker(job)
    if phase == "queued":
        async with exports.sessions() as session:
            await JobService(session).interrupt_job(job.id, MagicMock())
    elif phase == "before_export":
        process.cancel_event.set()
    else:
        export = exports.policy.export

        def interrupted(self, *args, **kwargs):
            export(self, *args, **kwargs)
            if phase == "failure":
                raise RuntimeError("export failed")
            process.cancel_event.set()

        monkeypatch.setattr(exports.policy, "export", interrupted)
    await process.run_loop()
    assert not exports.target.exists()
    assert not ModelExportService.staging(job.id).exists()
    async with exports.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == (
            JobStatus.FAILED if phase == "failure" else JobStatus.CANCELED
        )
    assert (await submit(exports)).id != job.id


async def test_completed_job_recovers_publication_without_changing_reads(exports, monkeypatch):
    job = await submit(exports)
    with monkeypatch.context() as patcher:
        patcher.setattr(ModelExportService, "publish", MagicMock(side_effect=RuntimeError("exited")))
        await worker(job).run_loop()
    assert not exports.target.exists()
    async with exports.sessions() as session:
        completed = await JobService(session).get_job_by_id(job.id)
        assert completed.status == JobStatus.COMPLETED
        with pytest.raises(DuplicateJobException):
            await submit(exports)
        handler = get_job_handler(job.type)
        assert handler.needs_exit_recovery(job, completed)
        await handler.recover(completed, JobService(session))
        assert not handler.needs_exit_recovery(job, completed)
    assert (exports.target / "model.pt").exists()
    ModelExportService.publish(exports.source, job.id, "torch")


def publication(statement):
    return (
        isinstance(statement, Update)
        and statement.table.name == "jobs"
        and statement.compile().params.get("status") == JobStatus.COMPLETED
    )


async def test_persisted_cancellation_wins_even_without_process_signal(exports, monkeypatch):
    job = await submit(exports)
    execute = AsyncSession.execute
    publishing, release = asyncio.Event(), asyncio.Event()

    async def pause(self, statement, *args, **kwargs):
        if publication(statement):
            publishing.set()
            await release.wait()
        return await execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", pause)
    task = asyncio.create_task(worker(job).run_loop())
    try:
        await asyncio.wait_for(publishing.wait(), 5)
        async with exports.sessions() as session:
            await JobService(session).interrupt_job(job.id, MagicMock())
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    assert not exports.target.exists()


@pytest.mark.parametrize("committed", [False, True])
async def test_commit_failures_keep_only_committed_output_recoverable(exports, monkeypatch, committed):
    job = await submit(exports)
    execute, commit = AsyncSession.execute, AsyncSession.commit

    async def track(self, statement, *args, **kwargs):
        if publication(statement):
            self.info["publishing"] = True
        return await execute(self, statement, *args, **kwargs)

    async def fail(self):
        if self.info.get("publishing") and not committed:
            raise RuntimeError("Commit failed")
        await commit(self)
        if self.info.get("publishing"):
            raise RuntimeError("Connection lost after commit")

    with monkeypatch.context() as patcher:
        patcher.setattr(AsyncSession, "execute", track)
        patcher.setattr(AsyncSession, "commit", fail)
        await worker(job).run_loop()
    assert not exports.target.exists()
    assert ModelExportService.staging(job.id).exists() == committed
    async with exports.sessions() as session:
        current = await JobService(session).get_job_by_id(job.id)
        assert current.status == (JobStatus.COMPLETED if committed else JobStatus.FAILED)
        if committed:
            await get_job_handler(job.type).recover(current, JobService(session))
            assert (exports.target / "model.pt").exists()


async def test_uncommitted_submission_releases_reservation(exports, monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(AsyncSession, "commit", AsyncMock(side_effect=RuntimeError("failed")))
        with pytest.raises(RuntimeError, match="failed"):
            await submit(exports)
    assert (await submit(exports)).payload.model_id == exports.source.id


async def test_model_deletion_is_blocked_by_active_export(exports):
    await submit(exports)
    assert (await exports.client.delete(f"/api/models/{exports.source.id}")).status_code == 409
    assert exports.root.exists()

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import multiprocessing as mp
import shutil
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql.dml import Update

from api.dependencies import get_scheduler
from db.engine import get_async_db_session
from db.schema import Base, ProjectDB
from exceptions import DuplicateJobException
from main import app
from schemas.base_job import JobStatus
from schemas.dataset import Dataset
from schemas.model import Model
from schemas.model_variant import ExportPackage
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
    monkeypatch.setattr("workers.model_optimization_worker.get_async_db_session_ctx", session_scope)
    monkeypatch.setitem(app.dependency_overrides, get_async_db_session, dependency)
    monkeypatch.setitem(app.dependency_overrides, get_settings, lambda: settings)
    scheduler = MagicMock()
    monkeypatch.setitem(app.dependency_overrides, get_scheduler, lambda: scheduler)
    source = Model(
        name="Test policy", path=str(models_dir / str(uuid4())), project_id=uuid4(), policy="act", properties={}
    )
    root = Path(source.path) / "exports/openvino"
    root.mkdir(parents=True)
    (root / "act.xml").write_text("policy graph")
    (root / "act.bin").write_bytes(b"original weights")
    (root / "tokenizer.xml").write_text("tokenizer graph")
    (root / "tokenizer.bin").write_bytes(b"tokenizer weights")
    (root / "manifest.json").write_text(
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
        session.add(ProjectDB(id=str(source.project_id), name="Test"))
        await session.commit()
        await ModelService(session).create_model(source)

    compressed = []

    def compress(path):
        compressed.append(Path(path))
        Path(path).with_suffix(".bin").write_bytes(b"INT8")

    monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    yield SimpleNamespace(
        sessions=session_scope,
        source=source,
        root=root,
        models_dir=models_dir,
        compressed=compressed,
        compress=compress,
        scheduler=scheduler,
    )
    await engine.dispose()


def worker(job):
    return ModelOptimizationWorker(job.id, mp.Event(), mp.Event(), MagicMock())


async def submit(context):
    async with context.sessions() as session:
        return await ModelOptimizationService(session).submit(context.source.id)


async def test_variant_preserves_model_identity_original_and_package_dependencies(optimization):
    before = {path.name: path.read_bytes() for path in optimization.root.iterdir()}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        url = f"/api/models/{optimization.source.id}/optimize"
        response = await client.post(url)
        assert response.status_code == 202
        assert (await client.post(url)).status_code == 409
        async with optimization.sessions() as session:
            job = await JobService(session).get_job_by_id(response.json()["id"])
        await worker(job).run_loop()
        detail = (await client.get(f"/api/models/{optimization.source.id}")).json()
        variants = detail["exports"][0]["variants"]
        assert [variant["id"] for variant in variants] == ["original", "int8"]
        assert variants[1]["weight_compression"] == "int8_sym"
        assert detail["model"]["id"] == str(optimization.source.id)
        assert (await client.get(f"/api/models/{job.id}")).status_code == 404
        assert (await client.post(url)).status_code == 409
    variant = optimization.root / "variants/int8"
    assert (variant / "act.bin").read_bytes() == b"INT8"
    assert (variant / "tokenizer.bin").read_bytes() == b"tokenizer weights"
    assert {name: (optimization.root / name).read_bytes() for name in before} == before
    package = ExportPackage(variant, "openvino", "int8")
    metadata = package.manifest
    assert metadata["studio"]["note"] == "preserved"
    assert metadata["other_extension"] == {"enabled": True}
    assert package.info.job_id == job.id
    async with optimization.sessions() as session:
        assert len(await ModelService(session).get_model_list()) == 1
        assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.COMPLETED


async def test_original_copy_does_not_copy_other_variants(optimization):
    sibling = optimization.root / "variants/unrelated"
    shutil.copytree(optimization.root, sibling, ignore=shutil.ignore_patterns("variants"))
    job = await submit(optimization)
    await worker(job).run_loop()
    assert not (optimization.root / "variants/int8/variants").exists()
    details = ModelService.get_backend_details(optimization.source)[0]
    assert details.size_bytes == next(item.size_bytes for item in details.variants if item.id == "original")
    assert details.size_bytes < sum(item.size_bytes for item in details.variants)


async def test_staging_is_outside_the_model_and_adds_no_hidden_paths(optimization, monkeypatch):
    def compress(path):
        optimization.compress(path)
        archive_path = ModelDownloadService().create_model_archive(Path(optimization.source.path))
        try:
            with zipfile.ZipFile(archive_path) as archive:
                assert not any("optimization" in name or "variants/int8" in name for name in archive.namelist())
        finally:
            archive_path.unlink()

    monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    job = await submit(optimization)
    await worker(job).run_loop()
    assert not ModelOptimizationService.staging(job.id).exists()
    assert not any(
        part.startswith(".")
        for path in optimization.models_dir.rglob("*")
        for part in path.relative_to(optimization.models_dir).parts
    )


@pytest.mark.parametrize("phase", ["queued", "before_compression", "during_compression", "failure"])
async def test_cancellation_and_failure_never_publish_and_can_retry(optimization, monkeypatch, phase):
    job = await submit(optimization)
    process = worker(job)
    if phase == "queued":
        async with optimization.sessions() as session:
            await JobService(session).interrupt_job(job.id, MagicMock())
    elif phase == "before_compression":
        process.cancel_event.set()
    else:

        def compress(path):
            optimization.compress(path)
            if phase == "failure":
                raise RuntimeError("compression failed")
            process.cancel_event.set()

        monkeypatch.setattr("physicalai.export.hooks.compress_weights_openvino_int8_sym", compress)
    await process.run_loop()
    async with optimization.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == (
            JobStatus.FAILED if phase == "failure" else JobStatus.CANCELED
        )
    assert not (optimization.root / "variants/int8").exists()
    assert not ModelOptimizationService.staging(job.id).exists()
    assert (optimization.root / "act.bin").read_bytes() == b"original weights"
    assert (await submit(optimization)).id != job.id


@pytest.mark.parametrize("artifact", ["missing.xml", "../outside.xml", "/tmp/outside.xml", "tokenizer.bin", None])
async def test_rejects_missing_unsafe_and_non_ir_policy_artifacts(optimization, artifact):
    path = optimization.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"]["artifacts"]["openvino"] = artifact
    path.write_text(json.dumps(manifest))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/api/models/{optimization.source.id}/optimize")
        assert response.status_code in {404, 409}


async def test_concurrent_submissions_reserve_only_one_target(optimization):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        results = await asyncio.gather(
            *(client.post(f"/api/models/{optimization.source.id}/optimize") for _ in range(3))
        )
    assert sorted(result.status_code for result in results) == [202, 409, 409]
    async with optimization.sessions() as session:
        assert len(await JobService(session).get_job_list()) == 1


async def test_selected_download_and_full_model_roundtrip(optimization, tmp_path, monkeypatch):
    job = await submit(optimization)
    await worker(job).run_loop()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for variant, weights in (("original", b"original weights"), ("int8", b"INT8")):
            response = await client.get(
                f"/api/models/{optimization.source.id}/exports/openvino/download", params={"variant": variant}
            )
            assert response.status_code == 200
            import io

            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                assert archive.read("act.bin") == weights
                assert not any(name.startswith("variants/") for name in archive.namelist())
    (Path(optimization.source.path) / "model.ckpt").write_text("checkpoint")
    archive_path = ModelDownloadService().create_model_archive(Path(optimization.source.path))
    restored = tmp_path / "restored"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(restored)
    archive_path.unlink()
    dataset = Dataset(
        name="Context", project_id=optimization.source.project_id, environment_id=uuid4(), default_task="test"
    )
    importer = ModelImportService(AsyncMock(return_value=dataset), AsyncMock(side_effect=lambda _job, model: model))
    monkeypatch.setattr(
        "services.model_import_service.get_settings", lambda: SimpleNamespace(models_dir=optimization.models_dir)
    )
    imported = await importer.import_model_directory(
        source_dir=restored, project_id=dataset.project_id, dataset_id=dataset.id, model_name="Restored"
    )
    assert (Path(imported.path) / "exports/openvino/variants/int8/act.bin").read_bytes() == b"INT8"
    assert len(ModelService.get_backend_details(imported)[0].variants) == 2


def publication(statement):
    return (
        isinstance(statement, Update)
        and statement.table.name == "jobs"
        and statement.compile().params.get("status") == JobStatus.COMPLETED
    )


async def test_cancellation_wins_before_publication_even_if_process_signal_is_lost(optimization, monkeypatch):
    job = await submit(optimization)
    execute = AsyncSession.execute
    publishing, release = asyncio.Event(), asyncio.Event()

    async def pause(self, statement, *args, **kwargs):
        if publication(statement):
            publishing.set()
            await release.wait()
        return await execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", pause)
    process = worker(job)
    task = asyncio.create_task(process.run_loop())
    try:
        await asyncio.wait_for(publishing.wait(), 5)
        async with optimization.sessions() as session:
            await JobService(session).interrupt_job(job.id, MagicMock())
        assert not process.cancel_event.is_set()
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    assert not (optimization.root / "variants/int8").exists()
    async with optimization.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.CANCELED


async def test_commit_failure_discards_staging_without_publishing(optimization, monkeypatch):
    job = await submit(optimization)
    execute, commit = AsyncSession.execute, AsyncSession.commit

    async def track(self, statement, *args, **kwargs):
        if publication(statement):
            self.info["publishing"] = True
        return await execute(self, statement, *args, **kwargs)

    async def fail(self):
        if self.info.get("publishing"):
            raise RuntimeError("Commit failed")
        await commit(self)

    monkeypatch.setattr(AsyncSession, "execute", track)
    monkeypatch.setattr(AsyncSession, "commit", fail)
    await worker(job).run_loop()
    assert not (optimization.root / "variants/int8").exists()
    assert not ModelOptimizationService.staging(job.id).exists()
    async with optimization.sessions() as session:
        assert (await JobService(session).get_job_by_id(job.id)).status == JobStatus.FAILED


async def test_uncertain_commit_keeps_output_for_publication_recovery(optimization, monkeypatch):
    from services.job_handlers import get_job_handler

    job = await submit(optimization)
    execute, commit = AsyncSession.execute, AsyncSession.commit

    async def track(self, statement, *args, **kwargs):
        if publication(statement):
            self.info["publishing"] = True
        return await execute(self, statement, *args, **kwargs)

    async def fail_after_commit(self):
        await commit(self)
        if self.info.get("publishing"):
            raise RuntimeError("Connection lost after commit")

    with monkeypatch.context() as patcher:
        patcher.setattr(AsyncSession, "execute", track)
        patcher.setattr(AsyncSession, "commit", fail_after_commit)
        await worker(job).run_loop()
    assert ModelOptimizationService.staging(job.id).exists()
    async with optimization.sessions() as session:
        completed = await JobService(session).get_job_by_id(job.id)
        assert completed.status == JobStatus.COMPLETED
        await get_job_handler(job.type).recover(completed, JobService(session))
    assert (optimization.root / "variants/int8/act.bin").read_bytes() == b"INT8"


async def test_completed_job_recovers_publication_without_changing_reads(optimization, monkeypatch):
    from services.job_handlers import get_job_handler

    job = await submit(optimization)
    publish = ModelOptimizationService.publish
    with monkeypatch.context() as patcher:
        patcher.setattr(ModelOptimizationService, "publish", MagicMock(side_effect=RuntimeError("worker exited")))
        await worker(job).run_loop()
    assert ("openvino", "int8") not in optimization.source.export_packages()
    async with optimization.sessions() as session:
        completed = await JobService(session).get_job_by_id(job.id)
        assert completed.status == JobStatus.COMPLETED
        model = await ModelService(session).get_model_by_id(optimization.source.id)
        assert ("openvino", "int8") not in model.export_packages()
        with pytest.raises(DuplicateJobException):
            await submit(optimization)
        handler = get_job_handler(job.type)
        assert handler.needs_exit_recovery(job, completed)
        await handler.recover(completed, JobService(session))
        assert not handler.needs_exit_recovery(job, completed)
    assert ("openvino", "int8") in model.export_packages()
    publish(model, job.id)  # Repeated recovery must not overwrite a published package.


async def test_orphan_recovery_only_removes_job_staging(optimization):
    job = await submit(optimization)
    async with optimization.sessions() as session:
        await JobService(session).claim_pending_job(job.id, job.type)
    staging = ModelOptimizationService.staging(job.id)
    ModelOptimizationService.compress(optimization.source, staging, job.id)
    destination = optimization.root / "variants/int8"
    async with optimization.sessions() as session:
        from services.job_handlers import get_job_handler

        await get_job_handler(job.type).recover(job, JobService(session))
    assert not destination.exists()
    assert not staging.exists()
    assert optimization.root.exists()
    assert (await submit(optimization)).id != job.id


async def test_uncommitted_submission_releases_its_database_reservation(optimization, monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(AsyncSession, "commit", AsyncMock(side_effect=RuntimeError("commit failed")))
        with pytest.raises(RuntimeError, match="commit failed"):
            await submit(optimization)
    assert (await submit(optimization)).payload.model_id == optimization.source.id


async def test_model_deletion_is_blocked_by_an_active_variant_job(optimization):
    await submit(optimization)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.delete(f"/api/models/{optimization.source.id}")).status_code == 409
    assert optimization.root.exists()


async def test_publication_that_wins_rejects_late_cancellation(optimization, monkeypatch):
    job = await submit(optimization)
    execute, commit = AsyncSession.execute, AsyncSession.commit
    publishing, canceling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def track(self, statement, *args, **kwargs):
        if publication(statement):
            self.info["publishing"] = True
        if isinstance(statement, Update) and statement.compile().params.get("status") == JobStatus.CANCELED:
            canceling.set()
        return await execute(self, statement, *args, **kwargs)

    async def pause(self):
        if self.info.get("publishing"):
            publishing.set()
            await release.wait()
        await commit(self)

    monkeypatch.setattr(AsyncSession, "execute", track)
    monkeypatch.setattr(AsyncSession, "commit", pause)
    process = asyncio.create_task(worker(job).run_loop())
    cancellation = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        try:
            await asyncio.wait_for(publishing.wait(), 5)
            detail = (await client.get(f"/api/models/{optimization.source.id}")).json()
            assert [item["id"] for item in detail["exports"][0]["variants"]] == ["original"]
            cancellation = asyncio.create_task(client.post(f"/api/jobs/{job.id}:interrupt"))
            await asyncio.wait_for(canceling.wait(), 5)
        finally:
            release.set()
            await asyncio.wait_for(process, 5)
            response = await asyncio.wait_for(cancellation, 5) if cancellation is not None else None
    assert response.status_code == 409
    assert (optimization.root / "variants/int8/act.bin").read_bytes() == b"INT8"

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from schemas.job import DatasetImportJob, Job, TrainJob
from services.log_service import LogService


class _StubJobService:
    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = jobs

    async def get_jobs_by_ids(self, _job_ids):
        return self.jobs


def _build_service(tmp_path, jobs: list[Job] | None = None) -> LogService:
    settings = SimpleNamespace(log_dir=tmp_path)
    return LogService(settings=settings, job_service=_StubJobService(jobs or []))


def test_resolve_source_path_for_static_and_job(tmp_path) -> None:
    service = _build_service(tmp_path)
    job_id = str(uuid4())

    assert service.resolve_source_path("application") == tmp_path / "app.log"
    job_log_path = service.resolve_source_path(f"job-{job_id}")
    assert job_log_path is not None
    assert job_log_path.name == f"{job_id}.log"
    assert job_log_path.parent.name == "jobs"
    assert service.resolve_source_path("does-not-exist") is None


@pytest.mark.anyio
async def test_discover_job_sources_includes_training_name_and_created_at(tmp_path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)

    job_id = uuid4()
    file_path = jobs_dir / f"{job_id}.log"
    file_path.write_text("hello")

    job = TrainJob(
        id=job_id,
        project_id=uuid4(),
        payload={
            "type": "training",
            "project_id": str(uuid4()),
            "dataset_id": str(uuid4()),
            "policy": "pi0",
            "model_name": "My Model",
            "max_steps": 100,
            "batch_size": 8,
            "base_model_id": None,
        },
        created_at=datetime.now(tz=UTC),
    )

    service = _build_service(tmp_path, jobs=[job])

    sources = await service.get_log_sources()
    job_sources = [source for source in sources if source.type == "job"]

    assert len(job_sources) == 1
    assert job_sources[0].id == f"job-{job_id}"
    assert job_sources[0].name == "My Model (pi0)"
    assert job_sources[0].created_at is not None


@pytest.mark.anyio
async def test_discover_job_sources_uses_staging_id_when_no_manifest(tmp_path) -> None:
    """New payload with archive_staging_id only -> display name derived from first 8 chars of staging id."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)

    job_id = uuid4()
    staging_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    file_path = jobs_dir / f"{job_id}.log"
    file_path.write_text("hello import staging")

    job = DatasetImportJob(
        id=job_id,
        project_id=uuid4(),
        payload={
            "type": "dataset_import",
            "step": "queued_for_detection",
            "archive_staging_id": staging_id,
            "format_hint": "auto",
            "dataset_manifest_draft": None,
            "validation_report": None,
            "finalize_input": None,
            "result_dataset_id": None,
        },
        created_at=datetime.now(tz=UTC),
    )

    service = _build_service(tmp_path, jobs=[job])

    sources = await service.get_log_sources()
    job_sources = [source for source in sources if source.type == "job"]

    assert len(job_sources) == 1
    assert job_sources[0].id == f"job-{job_id}"
    # First 8 chars of staging_id used as display name
    assert job_sources[0].name == "Import: aaaaaaaa.zip"


@pytest.mark.anyio
async def test_discover_job_sources_prefers_dataset_name(tmp_path) -> None:
    """Dataset name from prepare input takes priority for import job display names."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True)

    job_id = uuid4()
    staging_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    file_path = jobs_dir / f"{job_id}.log"
    file_path.write_text("hello import manifest")

    job = DatasetImportJob(
        id=job_id,
        project_id=uuid4(),
        payload={
            "type": "dataset_import",
            "step": "awaiting_user_review",
            "archive_staging_id": staging_id,
            "format_hint": "auto",
            "dataset_name": "My Preferred Dataset Name",
            "dataset_manifest_draft": {
                "source_type": "lerobot_v3",
                "capture": {},
                "schema": {},
            },
            "validation_report": None,
            "finalize_input": None,
            "result_dataset_id": None,
        },
        created_at=datetime.now(tz=UTC),
    )

    service = _build_service(tmp_path, jobs=[job])

    sources = await service.get_log_sources()
    job_sources = [source for source in sources if source.type == "job"]

    assert len(job_sources) == 1
    assert job_sources[0].name == "Import: My Preferred Dataset Name"


@pytest.mark.anyio
async def test_source_exists_is_false_for_missing_or_empty(tmp_path) -> None:
    service = _build_service(tmp_path)

    missing = tmp_path / "missing.log"
    empty = tmp_path / "empty.log"
    empty.write_text("")
    filled = tmp_path / "filled.log"
    filled.write_text("x")

    assert await service.source_exists(missing) is False
    assert await service.source_exists(empty) is False
    assert await service.source_exists(filled) is True

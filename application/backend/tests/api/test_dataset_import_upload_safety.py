import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dataset_import import _resolve_upload_size_estimate
from api.dependencies import get_dataset_import_service, get_job_service
from main import app
from schemas.base_job import JobStatus, JobType
from schemas.dataset_import_job import DatasetImportJobPayload, ImportStep
from schemas.job import DatasetImportJob
from settings import Settings

_FIXED_STAGING_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class _StubDatasetImportService:
    def __init__(self, project_id):
        self.project_id = project_id
        self.calls: list[dict] = []

    async def attach_dataset_import_archive(self, project_id, job_id, uploaded_archive_name):
        self.calls.append(
            {
                "project_id": project_id,
                "job_id": job_id,
                "uploaded_archive_name": uploaded_archive_name,
            }
        )
        return DatasetImportJob(
            id=job_id,
            project_id=project_id,
            status=JobStatus.PENDING,
            progress=5,
            message="Dataset queued for importing",
            payload=DatasetImportJobPayload(
                type=JobType.DATASET_IMPORT,
                step=ImportStep.QUEUED_FOR_DETECTION,
                archive_staging_id=_FIXED_STAGING_ID,
                uploaded_archive_name=uploaded_archive_name,
                format_hint="auto",
            ),
        )


class _StubJobService:
    """Minimal stub for JobService that returns a valid AWAITING_ARCHIVE_UPLOAD job."""

    def __init__(self, project_id, job_id):
        self.project_id = project_id
        self.job_id = job_id
        self._job = DatasetImportJob(
            id=job_id,
            project_id=project_id,
            status=JobStatus.PENDING,
            payload=DatasetImportJobPayload(
                step=ImportStep.AWAITING_ARCHIVE_UPLOAD,
                archive_staging_id=_FIXED_STAGING_ID,
                format_hint="auto",
            ),
        )

    async def get_job_by_id(self, job_id):
        return self._job


def _make_zip_bytes(files: dict[str, bytes], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=compression) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_upload_rejects_nested_zip_and_does_not_attach_archive() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    nested_zip_bytes = _make_zip_bytes({"payload.txt": b"hello"})
    outer_zip_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": b"parquet",
            "nested/payload.zip": nested_zip_bytes,
        }
    )

    # Use a small max_upload_bytes so disk-headroom check only requires a few MB,
    # making the test deterministic regardless of available disk space.
    with patch("api.dataset_import.get_settings") as mock_get_settings:
        settings = mock_get_settings.return_value
        settings.cache_dir = Settings().cache_dir
        settings.data_import_max_upload_bytes = 5 * 1024 * 1024  # 5 MB
        settings.data_import_min_free_bytes = 0
        settings.data_import_max_uncompressed_bytes = 10 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", outer_zip_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "nested zip entry" in body["message"]
    assert stub.calls == []


def test_upload_rejects_archive_with_too_large_uncompressed_size_and_does_not_attach_archive() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    large_payload = b"A" * 10_000
    archive_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": large_payload,
        },
        compression=zipfile.ZIP_STORED,
    )

    with patch("api.dataset_import.get_settings") as mock_get_settings:
        settings = mock_get_settings.return_value
        settings.cache_dir = Settings().cache_dir
        settings.data_import_max_upload_bytes = 5 * 1024 * 1024  # 5 MB - deterministic on CI
        settings.data_import_min_free_bytes = 0
        settings.data_import_max_uncompressed_bytes = 2_000

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "uncompressed size exceeds allowed limit" in body["message"]
    assert stub.calls == []


def test_upload_accepts_valid_zip_and_attaches_archive(tmp_path: Path) -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    archive_bytes = _make_zip_bytes(
        {
            "meta/info.json": b"{}",
            "data/episode.parquet": b"small-data",
        },
        compression=zipfile.ZIP_STORED,
    )

    staged_path = tmp_path / "cache" / "imports" / "datasets" / f"{_FIXED_STAGING_ID}.zip"
    with (
        patch("api.dataset_import.check_disk_headroom"),
        patch("api.dataset_import.resolve_payload_archive_path", return_value=staged_path),
        patch("api.dataset_import.get_settings") as mock_get_settings,
    ):
        settings = mock_get_settings.return_value
        settings.data_import_max_uncompressed_bytes = 10 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 202
    assert len(stub.calls) == 1
    assert stub.calls[0]["uploaded_archive_name"] == "dataset.zip"
    # The file should have been written to the staging path derived from the job's archive_staging_id.
    assert staged_path.exists()
    staged_path.unlink(missing_ok=True)


def test_upload_rejects_archive_with_too_many_entries() -> None:
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    archive_bytes = _make_zip_bytes(
        {
            **{f"data/file_{index}.txt": b"x" for index in range(101)},
            "meta/info.json": b"{}",
        },
        compression=zipfile.ZIP_STORED,
    )

    with (
        patch("api.dataset_import.get_settings") as mock_get_settings,
        patch("physicalai.data.archive_safety.DEFAULT_MAX_FILE_COUNT", 100),
    ):
        settings = mock_get_settings.return_value
        settings.cache_dir = Settings().cache_dir
        settings.data_import_max_upload_bytes = 5 * 1024 * 1024  # 5 MB - deterministic on CI
        settings.data_import_min_free_bytes = 0
        settings.data_import_max_uncompressed_bytes = 200 * 1024 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "zip_bomb_detected"
    assert "too many entries" in body["message"]
    assert stub.calls == []


# ---------------------------------------------------------------------------
# Large-upload guardrail tests
# ---------------------------------------------------------------------------


def test_upload_rejects_when_content_length_exceeds_max() -> None:
    """HTTP guard: reject before reading the body when Content-Length > limit."""
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    # A tiny but valid ZIP - the rejection must happen purely on the header value.
    archive_bytes = _make_zip_bytes(
        {"meta/info.json": b"{}"},
        compression=zipfile.ZIP_STORED,
    )

    # Patch settings so the threshold is lower than any real upload.
    # We set max_upload_bytes to 1 byte so the declared Content-Length (which
    # TestClient derives from the multipart body length) is always over the cap.
    with patch("middleware.upload_size_guard.get_settings") as mock_get_settings:
        settings = mock_get_settings.return_value
        settings.data_import_max_upload_bytes = 1  # 1 byte - always exceeded

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "upload_too_large"
    assert "exceeds" in body["message"]
    assert stub.calls == []


def test_upload_rejects_when_cache_dir_has_insufficient_free_space() -> None:
    """Disk guard: reject upload when cache dir has insufficient free space."""
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    archive_bytes = _make_zip_bytes(
        {"meta/info.json": b"{}"},
        compression=zipfile.ZIP_STORED,
    )

    import shutil

    # Build a fake disk_usage namedtuple that always reports zero free bytes.
    _fake_usage = shutil.disk_usage("/")._replace(free=0)

    with (
        patch("api.dataset_import.get_settings") as mock_get_settings,
        patch("physicalai.data.archive_safety.shutil.disk_usage", return_value=_fake_usage),
    ):
        settings = mock_get_settings.return_value
        settings.cache_dir = Settings().cache_dir
        settings.datasets_dir = Settings().datasets_dir
        settings.data_import_max_upload_bytes = 10 * 1024 * 1024 * 1024  # huge - no header rejection
        settings.data_import_min_free_bytes = 1  # any positive headroom will be unmet
        settings.data_import_max_uncompressed_bytes = 5 * 1024 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 507
    body = response.json()
    assert body["error_code"] == "insufficient_disk_space"
    assert stub.calls == []


def test_upload_disk_headroom_uses_actual_archive_size_not_configured_max() -> None:
    """Disk guard should use real upload size so large configured max does not over-reject."""
    project_id = uuid4()
    job_id = uuid4()
    stub = _StubDatasetImportService(project_id)
    job_stub = _StubJobService(project_id, job_id)
    app.dependency_overrides[get_dataset_import_service] = lambda: stub
    app.dependency_overrides[get_job_service] = lambda: job_stub

    archive_bytes = _make_zip_bytes(
        {"meta/info.json": b"{}"},
        compression=zipfile.ZIP_STORED,
    )

    import shutil

    min_free_bytes = 1024
    required_from_actual_size = len(archive_bytes)
    fake_free_bytes = required_from_actual_size + min_free_bytes
    _fake_usage = shutil.disk_usage("/")._replace(free=fake_free_bytes)

    with (
        patch("api.dataset_import.get_settings") as mock_get_settings,
        patch("physicalai.data.archive_safety.shutil.disk_usage", return_value=_fake_usage),
    ):
        settings = mock_get_settings.return_value
        settings.cache_dir = Settings().cache_dir
        settings.datasets_dir = Settings().datasets_dir
        settings.data_import_max_upload_bytes = 10 * 1024 * 1024 * 1024  # would fail if used for headroom
        settings.data_import_min_free_bytes = min_free_bytes
        settings.data_import_max_uncompressed_bytes = 5 * 1024 * 1024 * 1024

        try:
            client = TestClient(app)
            response = client.put(
                f"/api/projects/{project_id}/imports/datasets/{job_id}:upload",
                files={"archive": ("dataset.zip", archive_bytes, "application/zip")},
            )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 202
    assert len(stub.calls) == 1

    staged_path = Settings().cache_dir / "imports" / "datasets" / f"{_FIXED_STAGING_ID}.zip"
    staged_path.unlink(missing_ok=True)


def test_resolve_upload_size_estimate_falls_back_to_content_length() -> None:
    """Use Content-Length fallback when uploaded file object size cannot be read."""

    class _UnreadableUploadFile:
        def __init__(self) -> None:
            self.file = SimpleNamespace(tell=self._fail, seek=self._fail)

        @staticmethod
        def _fail(*_args, **_kwargs):
            raise OSError("cannot seek")

    archive = _UnreadableUploadFile()
    assert _resolve_upload_size_estimate(archive, "1234", fallback=9999) == 1234

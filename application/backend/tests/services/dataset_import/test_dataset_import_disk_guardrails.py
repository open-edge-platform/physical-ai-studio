"""Service-level tests for disk-headroom guardrails in the dataset import worker.

These tests exercise :func:`services.archive_safety.check_disk_headroom` and its
integration into :class:`workers.dataset_import_worker.DatasetImportWorker._run_commit`,
without performing any real disk I/O or network calls.
"""

import asyncio
import io
import shutil
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from exceptions import InsufficientDiskSpaceError
from services.archive_safety import check_disk_headroom

# ---------------------------------------------------------------------------
# Unit tests for the guard function itself
# ---------------------------------------------------------------------------


def test_check_disk_headroom_passes_when_free_space_is_sufficient(tmp_path: Path) -> None:
    """No exception is raised when there is ample free space."""
    _fake_usage = shutil.disk_usage("/")._replace(free=10 * 1024 * 1024 * 1024)  # 10 GiB

    with patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage):
        # Should not raise
        check_disk_headroom(tmp_path, required_bytes=1 * 1024 * 1024, min_free_bytes=1 * 1024 * 1024)


def test_check_disk_headroom_raises_when_free_space_is_insufficient(tmp_path: Path) -> None:
    """InsufficientDiskSpaceError is raised when free < required + headroom."""
    _fake_usage = shutil.disk_usage("/")._replace(free=0)

    with (
        patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage),
        pytest.raises(InsufficientDiskSpaceError) as exc_info,
    ):
        check_disk_headroom(tmp_path, required_bytes=1, min_free_bytes=1)

    assert "Insufficient disk space" in str(exc_info.value)
    assert exc_info.value.error_code == "insufficient_disk_space"
    assert exc_info.value.http_status == 507


def test_check_disk_headroom_passes_when_free_exactly_meets_need(tmp_path: Path) -> None:
    """Guard passes (does not raise) when free space exactly equals required + headroom."""
    required = 500 * 1024 * 1024  # 500 MiB
    headroom = 100 * 1024 * 1024  # 100 MiB
    _fake_usage = shutil.disk_usage("/")._replace(free=required + headroom)

    with patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage):
        # Exactly at the limit - should not raise
        check_disk_headroom(tmp_path, required_bytes=required, min_free_bytes=headroom)


def test_check_disk_headroom_raises_when_free_is_one_byte_short(tmp_path: Path) -> None:
    """Guard raises when free space is exactly one byte below the threshold."""
    required = 500 * 1024 * 1024
    headroom = 100 * 1024 * 1024
    _fake_usage = shutil.disk_usage("/")._replace(free=required + headroom - 1)

    with (
        patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage),
        pytest.raises(InsufficientDiskSpaceError),
    ):
        check_disk_headroom(tmp_path, required_bytes=required, min_free_bytes=headroom)


# ---------------------------------------------------------------------------
# Integration test: worker _run_commit raises when datasets dir is full
# ---------------------------------------------------------------------------


def test_worker_run_commit_raises_when_datasets_dir_has_insufficient_space(tmp_path: Path) -> None:
    """DatasetImportWorker._run_commit propagates InsufficientDiskSpaceError when
    the datasets directory filesystem reports no free space."""
    import multiprocessing as mp

    from schemas.dataset_import_job import (
        DatasetImportFinalizeInput,
        DatasetImportJobPayload,
        DatasetImportSource,
        DatasetManifest,
        ImportStep,
    )
    from workers.dataset_import_worker import DatasetImportWorker

    # --- set up a real archive file at the staging path so stat() succeeds ---
    staging_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    staging_dir = tmp_path / "imports" / "datasets"
    staging_dir.mkdir(parents=True)
    archive_path = staging_dir / f"{staging_id}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("meta/info.json", b"{}")
    archive_path.write_bytes(buf.getvalue())

    # --- build a minimal payload that has a committed manifest draft ---
    manifest_draft = DatasetManifest(source_type=DatasetImportSource.LEROBOT_V3)
    payload = DatasetImportJobPayload(
        step=ImportStep.QUEUED_FOR_IMPORT,
        archive_staging_id=staging_id,
        dataset_name="tmp",
        dataset_manifest_draft=manifest_draft,
        finalize_input=DatasetImportFinalizeInput(
            environment_id=uuid4(),
        ),
    )

    stop_event = mp.Event()
    event_queue: mp.Queue = mp.Queue()
    worker = DatasetImportWorker(stop_event=stop_event, event_queue=event_queue)
    worker.queue = MagicMock()

    _fake_usage = shutil.disk_usage("/")._replace(free=0)

    fake_settings = MagicMock()
    fake_settings.cache_dir = tmp_path
    fake_settings.datasets_dir = tmp_path / "datasets"
    fake_settings.data_import_min_free_bytes = 1  # any positive headroom is unmet with 0 free

    with (
        patch("services.dataset_import.adapters.lerobot_v3.get_settings", return_value=fake_settings),
        patch("services.dataset_import.staging.get_settings", return_value=fake_settings),
        patch("services.archive_safety.shutil.disk_usage", return_value=_fake_usage),
        patch("workers.dataset_import_worker.JobService.update_job_payload", new=AsyncMock(return_value=MagicMock())),
        pytest.raises(InsufficientDiskSpaceError),
    ):
        asyncio.run(
            worker._run_commit(
                job_id=uuid4(),
                project_id=uuid4(),
                payload=payload,
            )
        )

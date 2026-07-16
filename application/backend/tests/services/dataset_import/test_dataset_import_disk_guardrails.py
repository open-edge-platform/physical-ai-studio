"""Integration coverage for worker disk-headroom enforcement."""

import asyncio
import io
import multiprocessing as mp
import shutil
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from physicalai.data.archive_safety import InsufficientDiskSpaceError

from schemas.dataset_import_job import (
    DatasetImportFinalizeInput,
    DatasetImportJobPayload,
    DatasetImportSource,
    DatasetManifest,
    ImportStep,
)
from workers.dataset_import_worker import DatasetImportWorker


def test_worker_run_commit_raises_when_datasets_dir_has_insufficient_space(tmp_path: Path) -> None:
    """Propagate shared disk-space failures from the worker commit path."""
    staging_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    staging_dir = tmp_path / "imports" / "datasets"
    staging_dir.mkdir(parents=True)
    archive_path = staging_dir / f"{staging_id}.zip"
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("meta/info.json", b"{}")
    archive_path.write_bytes(archive_bytes.getvalue())

    payload = DatasetImportJobPayload(
        step=ImportStep.QUEUED_FOR_IMPORT,
        archive_staging_id=staging_id,
        dataset_name="tmp",
        dataset_manifest_draft=DatasetManifest(source_type=DatasetImportSource.LEROBOT_V3),
        finalize_input=DatasetImportFinalizeInput(environment_id=uuid4()),
    )
    worker = DatasetImportWorker(stop_event=mp.Event(), event_queue=mp.Queue())
    worker.queue = MagicMock()
    fake_settings = MagicMock(
        cache_dir=tmp_path,
        datasets_dir=tmp_path / "datasets",
        data_import_min_free_bytes=1,
    )
    disk_usage = shutil.disk_usage("/")._replace(free=0)

    with (
        patch("services.dataset_import.adapters.lerobot_v3.get_settings", return_value=fake_settings),
        patch("services.dataset_import.staging.get_settings", return_value=fake_settings),
        patch("physicalai.data.archive_safety.shutil.disk_usage", return_value=disk_usage),
        patch("workers.dataset_import_worker.JobService.update_job_payload", new=AsyncMock(return_value=MagicMock())),
        pytest.raises(InsufficientDiskSpaceError),
    ):
        asyncio.run(worker._run_commit(job_id=uuid4(), project_id=uuid4(), payload=payload))

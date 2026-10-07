# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-shot AWS Batch entry point.

Training itself is ``training.run_training_job`` and is tested there. What the
batch module owns is the S3 job layout, the status.json lifecycle, SIGTERM
cancellation, and the exit code.
"""

from __future__ import annotations

import json
import threading
import zipfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from trainer.batch import (
    EXIT_CANCELED,
    EXIT_FAILED,
    EXIT_OK,
    BatchJobStatus,
    BatchSettings,
    LocalDirJobStorage,
    StatusReporter,
    run,
)
from trainer.schemas import TrainerJobStatus
from training import TrainingJobSpec

if TYPE_CHECKING:
    from pathlib import Path

BATCH = "trainer.batch"


@pytest.fixture
def settings() -> BatchSettings:
    return BatchSettings(
        PHYSICALAI_BATCH_BUCKET="bucket",
        PHYSICALAI_BATCH_JOB_PREFIX="jobs/job-123",
        PHYSICALAI_BATCH_STATUS_INTERVAL_S=0,
    )


@pytest.fixture
def trainer_storage(tmp_path: Path):
    """Point the batch run and the archive helper at a throwaway storage directory."""
    trainer_settings = MagicMock()
    trainer_settings.storage_dir = tmp_path / "storage"
    trainer_settings.models_dir = tmp_path / "storage" / "models"
    trainer_settings.archives_dir = tmp_path / "storage" / "archives"
    trainer_settings.max_uncompressed_bytes = 1 << 30
    trainer_settings.min_free_bytes = 0
    with (
        patch(f"{BATCH}.get_settings", return_value=trainer_settings),
        patch("trainer.runner.get_settings", return_value=trainer_settings),
        patch("trainer.archive.get_settings", return_value=trainer_settings),
    ):
        yield trainer_settings


@pytest.fixture
def job_root(tmp_path: Path, settings: BatchSettings) -> Path:
    """A local stand-in for the job's S3 prefix with spec.json and dataset.zip."""
    root = tmp_path / "s3"
    root.mkdir()
    spec = TrainingJobSpec(policy="act", max_epochs=1)
    (root / settings.spec_key).write_text(spec.model_dump_json())
    with zipfile.ZipFile(root / settings.dataset_key, "w") as zf:
        # Two top-level entries so the single-root flattening does not rewrite paths.
        zf.writestr("meta/info.json", "{}")
        zf.writestr("data/chunk-000/episode_000000.parquet", b"")
    return root


def _status(root: Path, settings: BatchSettings) -> BatchJobStatus:
    return BatchJobStatus.model_validate_json((root / settings.status_key).read_text())


def _completed_run(_spec, *, output_dir, **_kwargs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.ckpt").write_bytes(b"weights")


def test_run_trains_from_the_fetched_dataset_and_uploads_the_artifact(
    trainer_storage, job_root: Path, settings: BatchSettings
) -> None:
    storage = LocalDirJobStorage(job_root, settings)
    seen: dict[str, object] = {}

    def _run(spec, *, dataset_root, output_dir, **_kwargs) -> None:
        # The work dir is removed after the run, so inspect it here.
        seen["policy"] = spec.policy
        seen["dataset_file"] = (dataset_root / "meta" / "info.json").is_file()
        seen["output_dir"] = output_dir
        _completed_run(spec, output_dir=output_dir)

    with patch("training.run_training_job", side_effect=_run):
        assert run(storage, settings) == EXIT_OK

    assert seen == {
        "policy": "act",
        "dataset_file": True,
        "output_dir": trainer_storage.models_dir / "job-123",
    }

    with zipfile.ZipFile(job_root / settings.artifact_key) as zf:
        assert zf.namelist() == ["model.ckpt"]
    assert _status(job_root, settings).status is TrainerJobStatus.COMPLETED
    assert _status(job_root, settings).progress == 100


def test_run_resets_run_options_from_the_serialised_spec(trainer_storage, job_root: Path, settings) -> None:
    """Studio's serialised run_options (e.g. a masked token) must not reach the run."""
    spec = TrainingJobSpec(policy="act", run_options={"hf_token": "**********", "resume_from": "x.ckpt"})
    (job_root / settings.spec_key).write_text(spec.model_dump_json())

    with patch("training.run_training_job", side_effect=_completed_run) as mock_run:
        run(LocalDirJobStorage(job_root, settings), settings)

    assert mock_run.call_args.args[0].run_options.hf_token is None
    assert mock_run.call_args.args[0].run_options.resume_from is None


def test_run_reports_canceled_when_stop_is_requested(trainer_storage, job_root: Path, settings) -> None:
    stop = threading.Event()
    stop.set()

    with patch("training.run_training_job"):
        assert run(LocalDirJobStorage(job_root, settings), settings, stop=stop) == EXIT_CANCELED

    assert _status(job_root, settings).status is TrainerJobStatus.CANCELED
    assert not (job_root / settings.artifact_key).exists()


def test_run_reports_failed_with_the_error_message(trainer_storage, job_root: Path, settings) -> None:
    with patch("training.run_training_job", side_effect=RuntimeError("CUDA OOM")):
        assert run(LocalDirJobStorage(job_root, settings), settings) == EXIT_FAILED

    status = _status(job_root, settings)
    assert status.status is TrainerJobStatus.FAILED
    assert status.message == "CUDA OOM"


def test_run_fails_on_an_invalid_dataset_archive(trainer_storage, job_root: Path, settings) -> None:
    (job_root / settings.dataset_key).write_bytes(b"not a zip")

    with patch("training.run_training_job") as mock_run:
        assert run(LocalDirJobStorage(job_root, settings), settings) == EXIT_FAILED

    mock_run.assert_not_called()
    assert _status(job_root, settings).status is TrainerJobStatus.FAILED


def test_status_reporter_throttles_progress_but_always_flushes_terminal_states() -> None:
    storage = MagicMock()
    clock = MagicMock(side_effect=[0.0, 1.0, 6.0, 6.5])
    reporter = StatusReporter(storage, interval_s=5.0, clock=clock)

    reporter.report(10, "a", None)
    reporter.report(20, "b", None)  # 1s later: dropped
    reporter.report(30, "c", None)  # 6s later: written
    reporter.finish(TrainerJobStatus.COMPLETED, progress=100)  # forced

    written = [call.args[0] for call in storage.put_status.call_args_list]
    assert [s.progress for s in written] == [10, 30, 100]
    assert written[-1].status is TrainerJobStatus.COMPLETED


def test_status_reporter_swallows_storage_errors() -> None:
    storage = MagicMock()
    storage.put_status.side_effect = OSError("s3 down")
    StatusReporter(storage, interval_s=0).report(1, None, None)


def test_batch_job_status_round_trips_through_json() -> None:
    status = BatchJobStatus(
        status=TrainerJobStatus.RUNNING, progress=42, message="m", updated_at="2026-01-01T00:00:00Z"
    )
    assert BatchJobStatus.model_validate(json.loads(status.model_dump_json())) == status


def test_settings_job_id_is_the_last_prefix_segment(settings: BatchSettings) -> None:
    assert settings.job_id == "job-123"


def test_main_rejects_an_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from trainer.batch.__main__ import main

    monkeypatch.setenv("PHYSICALAI_BATCH_PROVIDER", "nope")
    assert main() == 2


def test_main_wires_the_aws_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from trainer.batch import __main__ as entry

    monkeypatch.setenv("PHYSICALAI_BATCH_PROVIDER", "aws")
    monkeypatch.setenv("PHYSICALAI_BATCH_BUCKET", "bucket")
    monkeypatch.setenv("PHYSICALAI_BATCH_JOB_PREFIX", "jobs/x")
    with (
        patch("trainer.batch.s3.S3JobStorage") as storage_cls,
        patch.object(entry, "run", return_value=EXIT_OK) as run_mock,
        patch.object(entry, "install_sigterm_handler"),
    ):
        assert entry.main() == EXIT_OK

    assert run_mock.call_args.args[0] is storage_cls.return_value


def test_s3_storage_addresses_objects_under_the_job_prefix(tmp_path: Path, settings: BatchSettings) -> None:
    from trainer.batch.s3 import S3JobStorage

    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock(read=lambda: b'{"policy": "act"}')}
    storage = S3JobStorage(settings, client=client)

    assert storage.fetch_spec().policy == "act"
    client.get_object.assert_called_once_with(Bucket="bucket", Key="jobs/job-123/spec.json")

    storage.fetch_dataset(tmp_path / "in" / "dataset.zip")
    client.download_file.assert_called_once_with(
        "bucket", "jobs/job-123/dataset.zip", str(tmp_path / "in" / "dataset.zip")
    )

    storage.put_status(BatchJobStatus(status=TrainerJobStatus.RUNNING, updated_at="2026-01-01T00:00:00Z"))
    assert client.put_object.call_args.kwargs["Key"] == "jobs/job-123/status.json"

    storage.put_artifact(tmp_path / "artifact.zip")
    client.upload_file.assert_called_once_with(str(tmp_path / "artifact.zip"), "bucket", "jobs/job-123/artifact.zip")

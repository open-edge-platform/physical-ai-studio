# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Check the deployable wrapper's storage contract without a Modal account."""

import runpy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from trainer.batch.status import BatchJobStatus
from training import TrainingJobSpec


@pytest.mark.parametrize("job_id_is_valid", [True, False])
def test_modal_worker_commits_and_cleans_local_outputs(monkeypatch, tmp_path, job_id_is_valid):
    import sys

    volume = MagicMock()
    app = MagicMock()
    app.function.side_effect = lambda **kwargs: lambda function: function
    sdk = SimpleNamespace(App=MagicMock(return_value=app), Volume=MagicMock(), Image=MagicMock(), Secret=MagicMock())
    sdk.Volume.from_name.return_value = volume
    monkeypatch.setitem(sys.modules, "modal", sdk)
    worker = runpy.run_path(str(Path(__file__).parents[2] / "src" / "trainer" / "modal_app.py"))
    settings = SimpleNamespace(
        storage_dir=tmp_path / "local", models_dir=tmp_path / "models", archives_dir=tmp_path / "archives"
    )
    monkeypatch.setattr("trainer.settings.get_settings", lambda: settings)
    job_id = str(uuid4())
    remote_root = tmp_path / "remote"
    job_root = remote_root / "jobs" / job_id
    job_root.mkdir(parents=True)
    (job_root / "spec.json").write_text(TrainingJobSpec(policy="act").model_dump_json())
    output_dirs = [settings.models_dir / job_id, settings.storage_dir / "cache" / job_id]
    for directory in output_dirs:
        directory.mkdir(parents=True)
    settings.archives_dir.mkdir()
    archive = settings.archives_dir / f"{job_id}.zip"
    archive.write_bytes(b"export")

    def run(storage, batch_settings):
        assert batch_settings.job_id == job_id
        assert storage.fetch_spec().policy == "act"
        storage.put_artifact(archive)
        storage.put_status(BatchJobStatus(status="completed", progress=100, updated_at=datetime.now(UTC)))
        return 0

    monkeypatch.setattr("trainer.batch.run", run)
    worker["train"].__globals__["Path"] = lambda path: remote_root if path == "/training-jobs" else Path(path)
    if not job_id_is_valid:
        with pytest.raises(ValueError):
            worker["train"]("../not-a-job")
        volume.reload.assert_not_called()
        return
    assert worker["train"](job_id) == 0
    volume.reload.assert_called_once()
    assert volume.commit.call_count == 2
    assert (job_root / "artifact.zip").read_bytes() == b"export"
    assert (job_root / "status.json").exists()
    assert all(not directory.exists() for directory in output_dirs)
    assert not archive.exists()
    config = app.function.call_args.kwargs
    assert config["timeout"] == 24 * 60 * 60 and config["retries"] == 0
    assert config["volumes"] == {"/training-jobs": volume}

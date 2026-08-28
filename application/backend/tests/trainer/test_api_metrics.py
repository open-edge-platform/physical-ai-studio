# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trainer's live Lightning metrics stream."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trainer import api as api_module
from trainer.schemas import JobState, TrainerJobStatus


def test_metrics_stream_uses_studio_csv_shape(tmp_path, monkeypatch) -> None:
    job_id = str(uuid4())
    state = JobState(remote_job_id=job_id, status=TrainerJobStatus.COMPLETED)
    store = MagicMock()
    store.get.return_value = state
    settings = MagicMock()
    settings.storage_dir = tmp_path / "storage"
    settings.models_dir = tmp_path / "models"
    metrics_path = settings.models_dir / job_id / "version_0" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text("epoch,step,train/loss\n0,4,0.25\n", encoding="utf-8")
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)

    app = FastAPI()
    app.include_router(api_module.router)
    app.state.queue_manager = SimpleNamespace(store=store)

    response = TestClient(app).get(f"/jobs/{job_id}/metrics")

    assert response.status_code == 200
    data = next(line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: "))
    assert json.loads(data) == {"epoch": 0, "step": 4, "train_loss": 0.25}

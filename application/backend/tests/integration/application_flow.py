# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin API helpers shared by application end-to-end tests."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def wait_for_job(client: TestClient, job_id: str, *, step: str | None = None, timeout: float = 300) -> dict[str, Any]:
    """Poll the normal job endpoint while the real application workers execute."""
    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"failed", "canceled"}:
            raise AssertionError(f"Application job {job_id} ended as {job['status']}: {job.get('message')}")
        if (step is not None and job["payload"].get("step") == step) or (step is None and job["status"] == "completed"):
            return job
        time.sleep(0.25)
    raise AssertionError(f"Application job {job_id} timed out: {job.get('status')} / {job.get('message')}")


def wait_for_model(client: TestClient, project_id: str, job_id: str) -> dict[str, Any]:
    """Wait for model registration, which follows the terminal job-status write."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/projects/{project_id}/models")
        assert response.status_code == 200, response.text
        for model in response.json():
            if model.get("train_job_id") == job_id:
                return model
        time.sleep(0.1)
    raise AssertionError(f"Application completed job {job_id} without registering its model")

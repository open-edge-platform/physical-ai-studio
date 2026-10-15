# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from api.system import get_video_recording_presets, request_graceful_restart, restart_server
from main import app
from services.health_service import HealthService
from settings import get_settings


def test_get_system_info(monkeypatch) -> None:
    monkeypatch.setenv("STUDIO_GIT_SHA", "a" * 40)
    monkeypatch.setenv("STUDIO_GIT_COMMIT_DATE", "2026-10-09T09:46:34Z")
    monkeypatch.setenv("STUDIO_GIT_TAG", "app/v0.3.0")

    with TestClient(app) as client:
        response = client.get("/api/system/info")

    assert response.status_code == 200
    assert response.json() == {
        "studio_version": get_settings().version,
        "git_sha": "a" * 40,
        "git_commit_date": "2026-10-09T09:46:34Z",
        "git_tag": "app/v0.3.0",
        "git_dirty": False,
    }


async def test_restart_server_marks_restart_required_and_schedules_shutdown() -> None:
    background_tasks = BackgroundTasks()
    health_service = HealthService()

    response = await restart_server(background_tasks, health_service)

    assert response == {"status": "restarting"}
    assert health_service.plugin_restart_required is True
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is request_graceful_restart


async def test_get_video_recording_presets_uses_system_service() -> None:
    service = MagicMock()
    expected = MagicMock()
    service.get_video_recording_presets.return_value = expected

    assert await get_video_recording_presets(service) is expected

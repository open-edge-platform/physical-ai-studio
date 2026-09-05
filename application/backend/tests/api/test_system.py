import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from api.system import _stop_process, restart_server
from main import app
from schemas.health import HealthResponse
from services.health_service import HealthService


async def test_restart_server_marks_restart_required_and_schedules_shutdown() -> None:
    background_tasks = BackgroundTasks()
    health_service = HealthService()

    response = await restart_server(background_tasks, health_service)

    assert response == {"status": "restarting"}
    assert health_service.plugin_restart_required is True
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is _stop_process


@pytest.fixture
def health_client(monkeypatch) -> TestClient:
    # Drive the health endpoint without the full app lifespan: startup performs
    # hardware discovery and shutdown may exec a process replacement.
    app.state.health_service = HealthService()
    yield TestClient(app)
    app.state.health_service = None


def test_health_check_returns_typed_response(health_client: TestClient) -> None:
    response = health_client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert HealthResponse.model_validate(body) == HealthResponse(
        status="healthy",
        instance_id=body["instance_id"],
        restart_required=body["restart_required"],
    )
    assert body["instance_id"]
    assert body["restart_required"] is False


def test_health_check_reports_restart_required(health_client: TestClient) -> None:
    app.state.health_service.mark_plugin_restart_required()
    response = health_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["restart_required"] is True
    assert response.json()["instance_id"]

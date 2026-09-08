from fastapi.testclient import TestClient

from api.dependencies import get_health_service
from main import app
from services.health_service import HealthService


def test_health_endpoint_reports_instance_and_restart_state() -> None:
    health_service = HealthService()
    health_service.mark_plugin_restart_required()
    app.dependency_overrides[get_health_service] = lambda: health_service

    try:
        response = TestClient(app).get("/api/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "instance_id": health_service.instance_id,
        "restart_required": True,
    }
    assert response.headers["Cache-Control"] == "no-store"


def test_health_endpoint_reports_no_restart_by_default() -> None:
    health_service = HealthService()
    app.dependency_overrides[get_health_service] = lambda: health_service

    try:
        response = TestClient(app).get("/api/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["restart_required"] is False

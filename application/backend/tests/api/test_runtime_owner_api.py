from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_robot_service
from main import app


class _StubRobot:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubRobotService:
    def __init__(self, robot: _StubRobot) -> None:
        self.robot = robot
        self.deleted: list[UUID] = []

    async def get_robot_by_id(self, project_id: UUID, robot_id: UUID) -> _StubRobot:
        return self.robot

    async def delete_robot(self, project_id: UUID, robot_id: UUID) -> None:
        self.deleted.append(robot_id)


def _client(service: _StubRobotService) -> TestClient:
    app.dependency_overrides[get_robot_service] = lambda: service
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_runtime_locks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    xdg = tmp_path / "xdg-runtime"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    yield
    app.dependency_overrides.clear()


def test_deleting_a_held_robot_returns_423(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubRobotService(_StubRobot("left arm"))
    monkeypatch.setattr("api.robots.runtime_session_holder", lambda follower_id, timeout=1.0: {"pid": 41273})
    project_id = uuid4()
    robot_id = uuid4()

    response = _client(service).delete(f"/api/projects/{project_id}/robots/{robot_id}")

    assert response.status_code == 423
    body = response.json()
    assert body["error_code"] == "runtime_session_busy"
    assert "left arm" in body["message"]
    assert "41273" in body["message"]
    assert service.deleted == []


def test_deleting_a_robot_whose_lock_is_live_without_metadata_returns_423(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _StubRobotService(_StubRobot("left arm"))
    monkeypatch.setattr("runtime.owner.live_session_pid", lambda name: 41273)
    monkeypatch.setattr("runtime.owner.probe_session_metadata", lambda *args, **kwargs: None)
    project_id = uuid4()
    robot_id = uuid4()

    response = _client(service).delete(f"/api/projects/{project_id}/robots/{robot_id}")

    assert response.status_code == 423
    body = response.json()
    assert body["error_code"] == "runtime_session_busy"
    assert "41273" in body["message"]
    assert service.deleted == []


def test_deleting_an_unheld_robot_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _StubRobotService(_StubRobot("idle arm"))
    probe = MagicMock()
    monkeypatch.setattr("runtime.owner.probe_session_metadata", probe)
    project_id = uuid4()
    robot_id = uuid4()

    response = _client(service).delete(f"/api/projects/{project_id}/robots/{robot_id}")

    assert response.status_code == 204
    assert service.deleted == [robot_id]
    probe.assert_not_called()

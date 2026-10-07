from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_runtime_session_registry
from main import app
from runtime.contract import StateData
from runtime.handle import RuntimeProcessError
from runtime.ids import runtime_session_name
from runtime.registry import RuntimeSessionRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeHandle:
    """Just the surface RuntimeSessionService and the registry read."""

    def __init__(self, follower_id: UUID | None = None, **overrides: Any) -> None:
        self.follower_id = follower_id or uuid4()
        self.session_name = runtime_session_name(self.follower_id)
        self.follower_name: str | None = "left arm"
        self.leader_name: str | None = "left leader"
        self.camera_keys = ["overhead", "wrist"]
        self.started_at = datetime(2026, 3, 1, tzinfo=UTC)
        self.state: StateData | None = StateData(
            connected=True,
            follower_source="teleop",
            model_loaded=False,
            task="pick up the cube",
            dataset_loaded=True,
            is_recording=True,
            episodes_recorded=3,
        )
        self.error: RuntimeProcessError | None = None
        self.pid: int | None = 41273
        self.status = "running"
        self.stopping = False
        self.alive = True
        self.stops = 0
        self.survives_stop = False
        for key, value in overrides.items():
            setattr(self, key, value)

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.stops += 1
        self.stopping = True
        if not self.survives_stop:
            self.alive = False


@pytest.fixture
def registry() -> RuntimeSessionRegistry:
    return RuntimeSessionRegistry()


@pytest.fixture
def client(registry: RuntimeSessionRegistry) -> Iterator[TestClient]:
    app.dependency_overrides[get_runtime_session_registry] = lambda: registry
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _register(registry: RuntimeSessionRegistry, handle: _FakeHandle) -> _FakeHandle:
    asyncio.run(registry.acquire(handle))  # type: ignore[arg-type]
    return handle


def test_no_sessions_lists_nothing(client: TestClient) -> None:
    assert client.get("/api/runtime/sessions").json() == []
    assert client.get("/api/runtime/sessions/count").json() == {"count": 0}


def test_a_running_session_maps_through(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    handle = _register(registry, _FakeHandle())

    body = client.get("/api/runtime/sessions").json()

    assert len(body) == 1
    session = body[0]
    assert session["session_name"] == handle.session_name
    assert session["follower_id"] == str(handle.follower_id)
    assert session["status"] == "running"
    assert session["pid"] == 41273
    assert session["follower_name"] == "left arm"
    assert session["leader_name"] == "left leader"
    assert session["camera_keys"] == ["overhead", "wrist"]
    assert session["started_at"].startswith("2026-03-01")
    assert session["activity"] == {
        "connected": True,
        "follower_source": "teleop",
        "model_loaded": False,
        "task": "pick up the cube",
        "dataset_loaded": True,
        "is_recording": True,
        "episodes_recorded": 3,
    }
    assert session["error"] is None


def test_a_starting_session_has_no_activity_yet(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    _register(registry, _FakeHandle(state=None, status="starting"))

    session = client.get("/api/runtime/sessions").json()[0]

    assert session["status"] == "starting"
    assert session["activity"] is None


def test_a_fatal_session_reports_its_error(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    _register(registry, _FakeHandle(status="error", error=RuntimeProcessError("arm went away", "boom")))

    session = client.get("/api/runtime/sessions").json()[0]

    assert session["status"] == "error"
    assert session["error"] == {"message": "arm went away", "error_code": "boom"}


def test_sessions_are_listed_independently(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    """Running two arms at once is the normal case, not an edge case."""
    _register(registry, _FakeHandle(follower_name="left arm"))
    _register(registry, _FakeHandle(follower_name="right arm"))

    body = client.get("/api/runtime/sessions").json()

    assert client.get("/api/runtime/sessions/count").json() == {"count": 2}
    assert {session["follower_name"] for session in body} == {"left arm", "right arm"}


def test_stopping_a_session_stops_and_releases_it(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    handle = _register(registry, _FakeHandle())

    response = client.post(f"/api/runtime/sessions/{handle.session_name}/stop")

    assert response.status_code == 204
    assert handle.stops == 1
    assert registry.count() == 0


def test_stopping_an_unknown_session_is_a_no_op(client: TestClient) -> None:
    """Two browsers racing the same Stop must both succeed."""
    assert client.post(f"/api/runtime/sessions/{runtime_session_name(uuid4())}/stop").status_code == 204


@pytest.mark.parametrize("session_name", ["12345", "rt", "notrt-abc", "rt-x; kill 1"])
def test_a_name_that_is_not_a_session_is_rejected(client: TestClient, session_name: str) -> None:
    response = client.post(f"/api/runtime/sessions/{session_name}/stop")

    assert response.status_code == 422, f"{session_name!r} reached the route without being validated"
    assert response.json()["error_code"] == "invalid_runtime_session_name"


@pytest.mark.parametrize("session_name", ["../../etc/passwd", "rt-a/../../b"])
def test_a_name_with_slashes_never_reaches_the_handler(client: TestClient, session_name: str) -> None:
    assert client.post(f"/api/runtime/sessions/{session_name}/stop").status_code == 404


def test_a_session_that_survives_the_stop_is_reported(client: TestClient, registry: RuntimeSessionRegistry) -> None:
    """The endpoint confirms the process is gone rather than assuming it."""
    handle = _register(registry, _FakeHandle(survives_stop=True))

    response = client.post(f"/api/runtime/sessions/{handle.session_name}/stop")

    assert response.status_code == 500
    assert response.json()["error_code"] == "runtime_session_stop_failed"

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from main import app
from runtime.transport.ids import runtime_session_name
from runtime.transport.lock import SessionNameLock


@pytest.fixture(autouse=True)
def isolate_runtime_locks(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the lock directory at a private tmp dir.

    Sessions are discovered from the filesystem, so without this a developer's
    own running session would show up in these assertions.
    """
    xdg = tmp_path / "xdg-runtime"
    xdg.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg))
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _name() -> str:
    return runtime_session_name(uuid4())


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "protocol_version": 1,
        "status": "running",
        "pid": 41273,
        "identity_digest": "abc123",
        "camera_keys": ["wrist", "overhead"],
        "started_at": 1_772_000_000.0,
        "idle_timeout_s": 45.0,
        "follower_name": "left arm",
        "leader_name": "left leader",
        "attached": True,
        "idle_deadline": None,
        "state": {
            "event": "state",
            "data": {
                "connected": True,
                "follower_source": "teleop",
                "model_loaded": False,
                "task": "pick up the cube",
                "dataset_loaded": True,
                "is_recording": True,
                "episodes_recorded": 3,
            },
        },
    }
    metadata.update(overrides)
    return metadata


def _stub_probe(monkeypatch: pytest.MonkeyPatch, answers: dict[str, dict[str, Any] | None]) -> None:
    monkeypatch.setattr(
        "services.runtime_session_service.probe_session_metadata",
        lambda name, *args, **kwargs: answers.get(name),
    )


def test_no_sessions_lists_nothing(client: TestClient) -> None:
    assert client.get("/api/runtime/sessions").json() == []
    assert client.get("/api/runtime/sessions/count").json() == {"count": 0}


def test_a_held_lock_is_counted_without_probing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The footer polls the count on every page, so it must not open a transport session."""
    probes: list[str] = []
    monkeypatch.setattr(
        "services.runtime_session_service.probe_session_metadata",
        lambda name, *args, **kwargs: probes.append(name),
    )

    with SessionNameLock(_name()):
        assert client.get("/api/runtime/sessions/count").json() == {"count": 1}

    assert probes == []


def test_a_running_session_maps_its_metadata_through(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _name()
    _stub_probe(monkeypatch, {name: _metadata()})

    with SessionNameLock(name):
        body = client.get("/api/runtime/sessions").json()

    assert len(body) == 1
    session = body[0]
    assert session["session_name"] == name
    assert session["follower_id"] == name.removeprefix("rt-")
    assert session["status"] == "running"
    assert session["pid"] == 41273
    assert session["follower_name"] == "left arm"
    assert session["leader_name"] == "left leader"
    assert session["camera_keys"] == ["wrist", "overhead"]
    assert session["attached"] is True
    assert session["idle_deadline"] is None
    assert session["idle_timeout_s"] == 45.0
    assert session["started_at"].startswith("2026-")
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


def test_an_abandoned_session_reports_its_idle_deadline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _name()
    _stub_probe(monkeypatch, {name: _metadata(attached=False, idle_deadline=1_772_000_045.0)})

    with SessionNameLock(name):
        session = client.get("/api/runtime/sessions").json()[0]

    assert session["attached"] is False
    assert session["idle_deadline"] is not None


def test_a_session_that_will_not_answer_is_still_listed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Something is holding a robot and cannot say why. That is the row a user needs most."""
    name = _name()
    _stub_probe(monkeypatch, {})

    with SessionNameLock(name):
        session = client.get("/api/runtime/sessions").json()[0]

    assert session["session_name"] == name
    assert session["status"] == "unreachable"
    assert session["pid"] is not None
    assert session["follower_name"] is None
    assert session["activity"] is None
    assert session["camera_keys"] == []


def test_unrecognised_metadata_degrades_instead_of_failing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata comes from another process, so a bad payload must not take the list down."""
    name = _name()
    _stub_probe(
        monkeypatch,
        {
            name: _metadata(
                status="banana",
                pid="not-a-pid",
                camera_keys="wrist",
                started_at="yesterday",
                idle_timeout_s=None,
                attached="yes",
                state={"event": "state", "data": {"connected": True, "follower_source": "teleop", "extra": 1}},
            )
        },
    )

    with SessionNameLock(name):
        session = client.get("/api/runtime/sessions").json()[0]

    assert session["status"] == "unreachable"
    assert session["pid"] is not None  # fell back to the lock file
    assert session["camera_keys"] == []
    assert session["started_at"] is None
    assert session["idle_timeout_s"] is None
    assert session["attached"] is None
    assert "extra" not in session["activity"]


def test_a_session_that_has_not_published_state_yet_is_listed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A starting session has no "state" key at all -- the common path, not an edge."""
    name = _name()
    metadata = _metadata(status="starting")
    del metadata["state"]
    _stub_probe(monkeypatch, {name: metadata})

    with SessionNameLock(name):
        response = client.get("/api/runtime/sessions")

    assert response.status_code == 200
    assert response.json()[0]["status"] == "starting"
    assert response.json()[0]["activity"] is None


@pytest.mark.parametrize("state", ["running", 42, ["a"], None, {"data": "not-a-mapping"}])
def test_a_malformed_state_payload_does_not_break_the_listing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, state: Any
) -> None:
    """Indexing an event dict that turns out to be a string would take the whole list down."""
    name = _name()
    _stub_probe(monkeypatch, {name: _metadata(state=state)})

    with SessionNameLock(name):
        response = client.get("/api/runtime/sessions")

    assert response.status_code == 200
    assert response.json()[0]["activity"] is None


def test_a_fatal_session_reports_its_error(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _name()
    _stub_probe(
        monkeypatch,
        {name: _metadata(status="error", error={"event": "error", "message": "arm went away", "error_code": "boom"})},
    )

    with SessionNameLock(name):
        session = client.get("/api/runtime/sessions").json()[0]

    assert session["status"] == "error"
    assert session["error"] == {"message": "arm went away", "error_code": "boom"}


def test_sessions_are_listed_independently(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running two arms at once is the normal case, not an edge case."""
    first, second = _name(), _name()
    _stub_probe(monkeypatch, {first: _metadata(follower_name="left arm"), second: _metadata(follower_name="right arm")})

    with SessionNameLock(first), SessionNameLock(second):
        body = client.get("/api/runtime/sessions").json()
        assert client.get("/api/runtime/sessions/count").json() == {"count": 2}

    assert {session["follower_name"] for session in body} == {"left arm", "right arm"}


def test_stopping_a_session_signals_it(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _name()
    stopped: list[str] = []
    monkeypatch.setattr("services.runtime_session_service.stop_runtime_session", lambda n, **kwargs: stopped.append(n))

    response = client.post(f"/api/runtime/sessions/{name}/stop")

    assert response.status_code == 204
    assert stopped == [name]


def test_stopping_an_unknown_session_is_a_no_op(client: TestClient) -> None:
    """Two browsers racing the same Stop must both succeed."""
    assert client.post(f"/api/runtime/sessions/{_name()}/stop").status_code == 204


@pytest.mark.parametrize(
    "session_name",
    [
        "12345",
        "rt",
        "notrt-abc",
        "rt-x; kill 1",
    ],
)
def test_a_name_that_is_not_a_session_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_name: str
) -> None:
    """This validation is the only thing between a path parameter and a signalled pid."""
    stopped: list[str] = []
    monkeypatch.setattr("services.runtime_session_service.stop_runtime_session", lambda n, **kwargs: stopped.append(n))

    response = client.post(f"/api/runtime/sessions/{session_name}/stop")

    assert response.status_code == 422, f"{session_name!r} reached the route without being validated"
    assert response.json()["error_code"] == "invalid_runtime_session_name"
    assert stopped == []


@pytest.mark.parametrize("session_name", ["../../etc/passwd", "rt-a/../../b"])
def test_a_name_with_slashes_never_reaches_the_handler(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, session_name: str
) -> None:
    """Routing rejects these before validation gets a say. The signal must still never fire."""
    stopped: list[str] = []
    monkeypatch.setattr("services.runtime_session_service.stop_runtime_session", lambda n, **kwargs: stopped.append(n))

    response = client.post(f"/api/runtime/sessions/{session_name}/stop")

    assert response.status_code == 404
    assert stopped == []


def test_a_session_that_survives_the_stop_is_reported(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """stop_runtime_session swallows its signal errors, so the endpoint confirms rather than assumes."""
    name = _name()
    monkeypatch.setattr("services.runtime_session_service.stop_runtime_session", lambda n, **kwargs: None)

    with SessionNameLock(name):
        response = client.post(f"/api/runtime/sessions/{name}/stop")

    assert response.status_code == 500
    assert response.json()["error_code"] == "runtime_session_stop_failed"


def test_a_stale_lock_file_does_not_break_the_list(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_session_name admits names that are not UUIDs, so follower_id must tolerate one."""
    name = "rt-not-a-uuid"
    _stub_probe(monkeypatch, {name: _metadata()})

    with SessionNameLock(name):
        session = client.get("/api/runtime/sessions").json()[0]

    assert session["session_name"] == name
    assert session["follower_id"] is None

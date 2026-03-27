from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi.testclient import TestClient
from sse_starlette import ServerSentEvent

from api.dependencies import get_log_service
from main import app
from schemas.logs import LogSource


class _StubLogService:
    def __init__(self) -> None:
        self.sources = [
            LogSource(id="application", name="Application", type="application"),
            LogSource(id="job-123", name="Model A (pi0)", type="job"),
        ]
        self.paths: dict[str, Path | None] = {
            "application": Path("/tmp/app.log"),
            "empty": Path("/tmp/empty.log"),
            "unknown": None,
        }

    async def get_log_sources(self) -> list[LogSource]:
        return self.sources

    def resolve_source_path(self, source_id: str) -> Path | None:
        return self.paths.get(source_id)

    async def source_exists(self, path: Path) -> bool:
        return path.name != "empty.log"

    async def empty_log_stream(self) -> AsyncGenerator[ServerSentEvent]:
        yield ServerSentEvent(data="DONE")

    async def tail_log_file(self, _path: Path) -> AsyncGenerator[ServerSentEvent]:
        yield ServerSentEvent(data="line-1")


def test_get_log_sources_returns_expected_sources() -> None:
    app.dependency_overrides[get_log_service] = lambda: _StubLogService()

    try:
        client = TestClient(app)
        response = client.get("/api/logs/sources")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert [source["id"] for source in body] == ["application", "job-123"]
    assert [source["type"] for source in body] == ["application", "job"]


def test_stream_logs_returns_empty_stream_for_unknown_source() -> None:
    app.dependency_overrides[get_log_service] = lambda: _StubLogService()

    try:
        client = TestClient(app)
        response = client.get("/api/logs/unknown/stream")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "data: DONE" in response.text


def test_stream_logs_returns_done_for_empty_source() -> None:
    app.dependency_overrides[get_log_service] = lambda: _StubLogService()

    try:
        client = TestClient(app)
        response = client.get("/api/logs/empty/stream")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "data: DONE" in response.text


def test_stream_logs_streams_lines_for_existing_source() -> None:
    app.dependency_overrides[get_log_service] = lambda: _StubLogService()

    try:
        client = TestClient(app)
        response = client.get("/api/logs/application/stream")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "data: line-1" in response.text

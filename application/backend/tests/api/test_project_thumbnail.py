from dataclasses import dataclass
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from api.dependencies import get_project_service, get_project_thumbnail_service
from main import app
from schemas import Dataset, Project


@dataclass
class _StubThumbnail:
    content: bytes
    etag: str
    last_modified: str


class _StubProjectService:
    def __init__(self, project: Project) -> None:
        self._project = project

    async def get_project_by_id(self, project_id: UUID) -> Project:
        return self._project


class _StubProjectThumbnailService:
    def __init__(self, thumbnail: _StubThumbnail | None) -> None:
        self._thumbnail = thumbnail
        self.calls: list[dict[str, object]] = []

    def get_thumbnail(
        self,
        project: Project,
        width: int = 320,
        height: int = 240,
    ) -> _StubThumbnail | None:
        self.calls.append(
            {
                "project": project,
                "width": width,
                "height": height,
            }
        )
        return self._thumbnail


def _make_dataset() -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Dataset 1",
        default_task="Task",
        project_id=uuid4(),
        environment_id=uuid4(),
    )


def _make_project(datasets: list[Dataset]) -> Project:
    return Project(
        id=uuid4(),
        name="Project",
        datasets=datasets,
    )


def test_project_thumbnail_returns_png() -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    thumbnail_service = _StubProjectThumbnailService(
        thumbnail=_StubThumbnail(
            content=b"png-bytes",
            etag='"project-etag"',
            last_modified="Wed, 06 Jan 2026 10:00:00 GMT",
        )
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_project_thumbnail_service] = lambda: thumbnail_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail?width=640&height=360")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png-bytes"
    assert thumbnail_service.calls == [
        {
            "project": project,
            "width": 640,
            "height": 360,
        }
    ]


def test_project_thumbnail_ignores_conditional_headers() -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    thumbnail_service = _StubProjectThumbnailService(
        thumbnail=_StubThumbnail(
            content=b"png-bytes",
            etag='"project-etag"',
            last_modified="Wed, 06 Jan 2026 10:00:00 GMT",
        )
    )

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_project_thumbnail_service] = lambda: thumbnail_service

    try:
        client = TestClient(app)
        response = client.get(
            f"/api/projects/{project.id}/thumbnail",
            headers={
                "If-None-Match": '"project-etag"',
                "If-Modified-Since": "Wed, 06 Jan 2026 10:00:00 GMT",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200


def test_project_thumbnail_returns_404_without_datasets() -> None:
    project = _make_project([])
    thumbnail_service = _StubProjectThumbnailService(thumbnail=None)

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_project_thumbnail_service] = lambda: thumbnail_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_project_thumbnail_returns_404_without_thumbnail() -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    thumbnail_service = _StubProjectThumbnailService(thumbnail=None)

    app.dependency_overrides[get_project_service] = lambda: _StubProjectService(project)
    app.dependency_overrides[get_project_thumbnail_service] = lambda: thumbnail_service

    try:
        client = TestClient(app)
        response = client.get(f"/api/projects/{project.id}/thumbnail")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404

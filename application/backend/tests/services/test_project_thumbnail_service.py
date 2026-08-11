from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from schemas import Dataset, EpisodeInfo, Project
from services.episode_thumbnail_service import EpisodeThumbnail
from services.project_thumbnail_service import ProjectThumbnailService


def _make_dataset() -> Dataset:
    return Dataset(
        id=uuid4(),
        name="Dataset 1",
        default_task="Task",
        project_id=uuid4(),
        environment_id=uuid4(),
    )


def _make_project(datasets: list[Dataset]) -> Project:
    return Project(id=uuid4(), name="Project", datasets=datasets)


def test_get_thumbnail_returns_none_without_datasets() -> None:
    episode_thumbnail_service = MagicMock()
    service = ProjectThumbnailService(episode_thumbnail_service=episode_thumbnail_service)

    thumbnail = service.get_thumbnail(project=_make_project([]), width=320, height=240)

    assert thumbnail is None
    episode_thumbnail_service.get_thumbnail.assert_not_called()


def test_get_thumbnail_returns_none_without_episodes(monkeypatch: pytest.MonkeyPatch) -> None:
    episode_thumbnail_service = MagicMock()
    service = ProjectThumbnailService(episode_thumbnail_service=episode_thumbnail_service)
    dataset_client = MagicMock()
    dataset_client.get_episode_infos.return_value = []
    monkeypatch.setattr("services.project_thumbnail_service.get_internal_read_dataset", lambda _dataset: dataset_client)

    thumbnail = service.get_thumbnail(project=_make_project([_make_dataset()]), width=320, height=240)

    assert thumbnail is None
    episode_thumbnail_service.get_thumbnail.assert_not_called()


def test_get_thumbnail_delegates_to_episode_thumbnail_service(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset()
    project = _make_project([dataset])
    dataset_client = MagicMock()
    dataset_client.get_episode_infos.return_value = [EpisodeInfo(episode_index=3, tasks=["task"], length=20, fps=30)]
    monkeypatch.setattr("services.project_thumbnail_service.get_internal_read_dataset", lambda _dataset: dataset_client)

    expected = EpisodeThumbnail(
        content=b"png",
        etag='"etag"',
        last_modified="Wed, 06 Jan 2026 10:00:00 GMT",
    )
    episode_thumbnail_service = MagicMock()
    episode_thumbnail_service.get_thumbnail.return_value = expected
    service = ProjectThumbnailService(episode_thumbnail_service=episode_thumbnail_service)

    thumbnail = service.get_thumbnail(project=project, width=156, height=156)

    assert thumbnail == expected
    episode_thumbnail_service.get_thumbnail.assert_called_once_with(
        dataset_id=dataset.id,
        dataset=dataset_client,
        episode_index=3,
        width=156,
        height=156,
    )

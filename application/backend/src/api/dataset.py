import asyncio
import os
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints
from starlette.background import BackgroundTask

from api.dependencies import (
    RuntimeSessionRegistryDep,
    get_dataset_download_service,
    get_dataset_service,
    get_episode_thumbnail_service,
)
from api.utils import safe_archive_name
from exceptions import ResourceInUseError, ResourceType
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.delete_episode_mutation import DeleteEpisodesMutation
from internal_datasets.mutations.edit_episode_task_mutation import EditEpisodeTaskMutation
from internal_datasets.utils import get_internal_read_dataset
from runtime_session.dataset import editing, is_being_edited
from schemas import Dataset, Episode, EpisodeInfo
from services import DatasetDownloadService, DatasetService, EpisodeThumbnailService

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: UUID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Dataset:
    """Get dataset by id"""
    return await dataset_service.get_dataset_by_id(dataset_id)


@router.get("/{dataset_id}/episodes")
async def get_episodes_of_dataset(
    dataset_id: UUID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[EpisodeInfo]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_read_dataset(dataset)
    return internal_dataset.get_episode_infos()


@router.get("/{dataset_id}/episodes/{episode_index}")
async def get_single_episode_of_dataset(
    dataset_id: UUID,
    episode_index: int,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Episode | None:
    """Get one dataset episode by index."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_read_dataset(dataset)
    return internal_dataset.find_episode(episode_index)


class EpisodeTaskUpdate(BaseModel):
    task: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@router.patch("/{dataset_id}/episodes/{episode_index}")
async def update_episode_task(
    dataset_id: UUID,
    episode_index: int,
    payload: EpisodeTaskUpdate,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    registry: RuntimeSessionRegistryDep,
) -> Episode:
    """Change the task an episode was recorded with."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    if (holder := registry.holder_of_dataset(dataset.id)) is not None:
        # The session records into a copy of the dataset and copies it back when it unloads, undoing the edit.
        raise ResourceInUseError(
            ResourceType.DATASET,
            dataset.id,
            message=f"Dataset {dataset.name!r} is loaded in {holder}. Unload it there, then try again.",
        )
    if is_being_edited(dataset.id):
        raise ResourceInUseError(
            ResourceType.DATASET,
            dataset.id,
            message=f"Dataset {dataset.name!r} is being edited. Try again in a moment.",
        )
    # Checked and marked on the event loop, so no recording opens the dataset and no other edit starts until it is done.
    with editing(dataset.id):
        episode = await asyncio.to_thread(
            lambda: EditEpisodeTaskMutation(get_internal_read_dataset(dataset)).edit_task(episode_index, payload.task)
        )
    if episode is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Episode {episode_index} not found")
    return episode


@router.get("/{dataset_id}/episodes/{episode_index}/thumbnail")
async def get_episode_thumbnail(  # noqa: PLR0913
    dataset_id: UUID,
    episode_index: int,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    thumbnail_service: Annotated[EpisodeThumbnailService, Depends(get_episode_thumbnail_service)],
    request: Request,
    camera: str | None = None,
    width: Annotated[int, Query(ge=32, le=1920)] = 320,
    height: Annotated[int, Query(ge=32, le=1080)] = 240,
) -> Response:
    """Get a thumbnail image for one episode."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_read_dataset(dataset)

    if not isinstance(internal_dataset, InternalLeRobotDataset):
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Thumbnail is unsupported")

    thumbnail = thumbnail_service.get_thumbnail(
        dataset_id=dataset_id,
        dataset=internal_dataset,
        episode_index=episode_index,
        camera=camera,
        width=width,
        height=height,
    )

    if thumbnail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Episode thumbnail not found")

    cache_headers = {
        "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
        "ETag": thumbnail.etag,
        "Last-Modified": thumbnail.last_modified,
        "Vary": "Accept",
    }

    request_etag = request.headers.get("if-none-match")
    request_last_modified = request.headers.get("if-modified-since")
    if request_etag == thumbnail.etag or request_last_modified == thumbnail.last_modified:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=cache_headers)

    return Response(content=thumbnail.content, media_type="image/png", headers=cache_headers)


@router.delete("/{dataset_id}/episodes")
async def delete_episodes_of_dataset(
    dataset_id: UUID,
    episode_indices: list[int],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    dataset_client = get_internal_read_dataset(dataset)
    mutation = DeleteEpisodesMutation(dataset_client)
    result = mutation.delete_episodes(episode_indices)
    return result.get_episodes()


@router.get("/{dataset_id}/video/{video_path:path}")
async def dataset_video_endpoint(
    dataset_id: UUID,
    video_path: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get path to video of episode"""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    dataset_base = Path(dataset.path).resolve()

    normalized_video_path = Path(video_path)
    if normalized_video_path.is_absolute() or ".." in normalized_video_path.parts:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    requested_path = os.path.realpath(dataset_base / normalized_video_path)

    # Include the separator so sibling directories sharing the dataset prefix are excluded.
    if not requested_path.startswith(f"{dataset_base}{os.sep}"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    if not Path(requested_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    # Recording appends episodes to the same mp4, so a cached copy goes stale: it ends where the
    # file ended when it was fetched, and every later episode then fails to play. Make browsers
    # check the ETag before reusing it.
    return FileResponse(requested_path, headers={"Cache-Control": "no-cache"})


@router.get("/{dataset_id}/download")
async def dataset_download_endpoint(
    dataset_id: UUID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    dataset_download_service: Annotated[DatasetDownloadService, Depends(get_dataset_download_service)],
) -> FileResponse:
    """Download dataset folder as a zip archive."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    dataset_path = Path(dataset.path).resolve()

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset path not found.")

    archive_path = await asyncio.to_thread(dataset_download_service.create_dataset_archive, dataset_path)
    filename = f"{safe_archive_name(dataset.name, fallback='dataset')}.zip"
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: Dataset, dataset_service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Dataset:
    """Create a new dataset."""
    return await dataset_service.create_dataset(dataset)


class DatasetNameUpdate(BaseModel):
    name: str


@router.put("/{dataset_id}")
async def update_dataset_name(
    dataset_id: UUID,
    payload: DatasetNameUpdate,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Dataset:
    """Update dataset name by id."""
    return await dataset_service.update_dataset_name(dataset_id=dataset_id, name=payload.name)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    remove_files: bool = False,
) -> None:
    """Delete dataset by id and optionally remove dataset files."""
    await dataset_service.delete_dataset(dataset_id=dataset_id, remove_files=remove_files)

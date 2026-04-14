from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from api.dependencies import (
    HTTPException,
    get_dataset_download_service,
    get_dataset_id,
    get_dataset_service,
    get_episode_thumbnail_service,
)
from api.utils import safe_archive_name
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from internal_datasets.mutations.delete_episode_mutation import DeleteEpisodesMutation
from internal_datasets.utils import get_internal_dataset
from schemas import Dataset, Episode, EpisodeInfo
from services import DatasetDownloadService, DatasetService, EpisodeThumbnailService

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Dataset:
    """Get dataset by id"""
    return await dataset_service.get_dataset_by_id(dataset_id)


@router.get("/{dataset_id}/episodes")
async def get_episodes_of_dataset(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[EpisodeInfo]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_dataset(dataset)
    return internal_dataset.get_episode_infos()


@router.get("/{dataset_id}/episodes/{episode_index}")
async def get_single_episode_of_dataset(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    episode_index: int,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Episode | None:
    """Get one dataset episode by index."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_dataset(dataset)
    return internal_dataset.find_episode(episode_index)


@router.get("/{dataset_id}/episodes/{episode_index}/thumbnail")
async def get_episode_thumbnail(  # noqa: PLR0913
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
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
    internal_dataset = get_internal_dataset(dataset)

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
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    episode_indices: list[int],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    dataset_client = get_internal_dataset(dataset)
    mutation = DeleteEpisodesMutation(dataset_client)
    result = mutation.delete_episodes(episode_indices)
    return result.get_episodes()


@router.get("/{dataset_id}/video/{video_path:path}")
async def dataset_video_endpoint(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    video_path: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get path to video of episode"""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    requested_path = (Path(dataset.path) / video_path).resolve()

    if not str(requested_path).startswith(str(dataset.path)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to the requested file is forbidden.")

    if not requested_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found.")

    return FileResponse(requested_path)


@router.get("/{dataset_id}/download")
async def dataset_download_endpoint(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    dataset_download_service: Annotated[DatasetDownloadService, Depends(get_dataset_download_service)],
) -> FileResponse:
    """Download dataset folder as a zip archive."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    dataset_path = Path(dataset.path).resolve()

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset path not found.")

    archive_path = dataset_download_service.create_dataset_archive(dataset_path)
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
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    payload: DatasetNameUpdate,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> Dataset:
    """Update dataset name by id."""
    return await dataset_service.update_dataset_name(dataset_id=dataset_id, name=payload.name)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: Annotated[UUID, Depends(get_dataset_id)],
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
    remove_files: bool = False,
) -> None:
    """Delete dataset by id and optionally remove dataset files."""
    await dataset_service.delete_dataset(dataset_id=dataset_id, remove_files=remove_files)

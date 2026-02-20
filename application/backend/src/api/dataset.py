from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse

from api.dependencies import HTTPException, get_dataset_id, get_dataset_service
from internal_datasets.mutations.delete_episode_mutation import DeleteEpisodesMutation
from internal_datasets.utils import get_internal_dataset
from schemas import Dataset, Episode
from services import DatasetService

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
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(dataset_id)
    internal_dataset = get_internal_dataset(dataset)
    return internal_dataset.get_episodes()


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


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: Dataset, dataset_service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Dataset:
    """Create a new dataset."""
    return await dataset_service.create_dataset(dataset)

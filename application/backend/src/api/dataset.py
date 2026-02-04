from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse

from api.dependencies import get_dataset_service
from internal_datasets.utils import get_internal_dataset
from schemas import Dataset, Episode
from services import DatasetService

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/{dataset_id}/episodes")
async def get_episodes_of_dataset(
    dataset_id: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(UUID(dataset_id))
    internal_dataset = get_internal_dataset(dataset)
    return internal_dataset.get_episodes()


@router.get("/{dataset_id}/{episode}/{camera}.mp4")
async def dataset_video_endpoint(
    dataset_id: str,
    episode: int,
    camera: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get path to video of episode"""
    dataset = await dataset_service.get_dataset_by_id(UUID(dataset_id))
    internal_dataset = get_internal_dataset(dataset)
    video_path = internal_dataset.get_video_path(episode, camera)
    return FileResponse(video_path)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: Dataset, dataset_service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Dataset:
    """Create a new dataset."""
    return await dataset_service.create_dataset(dataset)

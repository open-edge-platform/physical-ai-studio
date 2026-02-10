from settings import get_settings
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse

from api.dependencies import get_dataset_service, HTTPException
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


@router.get("/video/{video_path:path}")
async def dataset_video_endpoint(
    video_path: str,
) -> FileResponse:
    """Get path to video of episode"""
    settings = get_settings()
    requested_path = (settings.datasets_dir / video_path).resolve()

    # Verify that the resolved path is still under the base directory
    if not str(requested_path).startswith(str(settings.datasets_dir)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to the requested file is forbidden."
        )

    # Optional: confirm the file exists and is a regular file
    if not requested_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found."
        )

    return FileResponse(requested_path)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: Dataset, dataset_service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Dataset:
    """Create a new dataset."""
    return await dataset_service.create_dataset(dataset)

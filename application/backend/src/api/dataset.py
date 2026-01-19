from uuid import UUID
import os
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from api.dependencies import get_dataset_service
from schemas import Dataset, Episode, LeRobotDatasetInfo
from services import DatasetService
from utils.dataset import get_dataset_episodes, get_geti_action_datasets, get_local_repositories

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


@router.get("/lerobot")
async def list_le_robot_datasets() -> list[LeRobotDatasetInfo]:
    """Get all local lerobot datasets from huggingface cache."""
    return [
        LeRobotDatasetInfo(
            root=str(dataset.root),
            repo_id=dataset.repo_id,
            total_episodes=dataset.total_episodes,
            total_frames=dataset.total_frames,
            fps=dataset.fps,
            features=list(dataset.features),
            robot_type=dataset.robot_type,
        )
        for dataset in get_local_repositories()
    ]


@router.get("/orphans")
async def list_orphan_datasets(
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[LeRobotDatasetInfo]:
    """Get all local lerobot datasets from huggingface cache."""
    datasets = [dataset.path for dataset in await dataset_service.get_dataset_list()]

    return [
        LeRobotDatasetInfo(
            root=str(dataset.root),
            repo_id=dataset.repo_id,
            total_episodes=dataset.total_episodes,
            total_frames=dataset.total_frames,
            fps=dataset.fps,
            features=list(dataset.features),
            robot_type=dataset.robot_type,
        )
        for dataset in [
            LeRobotDatasetMetadata(repo, root=root) for repo, root in get_geti_action_datasets() if root not in datasets
        ]
    ]


@router.get("/{dataset_id}/episodes")
async def get_episodes_of_dataset(
    dataset_id: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = await dataset_service.get_dataset_by_id(UUID(dataset_id))
    return get_dataset_episodes(dataset.path)


@router.get("/{dataset_id}/{episode}/{camera}.mp4")
async def dataset_video_endpoint(
    dataset_id: str,
    episode: int,
    camera: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> FileResponse:
    """Get path to video of episode"""
    dataset = await dataset_service.get_dataset_by_id(UUID(dataset_id))
    metadata = LeRobotDatasetMetadata(dataset.name, dataset.path, None, force_cache_sync=False)
    full_camera_name = f"observation.images.{camera}"
    video_path = os.path.join(metadata.root, metadata.get_video_file_path(episode, full_camera_name))
    return FileResponse(video_path)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_dataset(
    dataset: Dataset, dataset_service: Annotated[DatasetService, Depends(get_dataset_service)]
) -> Dataset:
    """Create a new dataset."""
    return await dataset_service.create_dataset(dataset)

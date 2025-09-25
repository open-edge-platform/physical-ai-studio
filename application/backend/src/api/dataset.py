from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_dataset_service
from schemas import Episode, LeRobotDatasetInfo
from services import DatasetService
from utils.dataset import get_dataset_episodes, get_local_repositories

router = APIRouter()


@router.get("/lerobot_datasets")
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


@router.get("/{dataset_id}/episodes")
async def get_episodes_of_dataset(
    dataset_id: str,
    dataset_service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[Episode]:
    """Get dataset episodes of dataset by id."""
    dataset = dataset_service.get_dataset_by_id(dataset_id)
    return get_dataset_episodes(dataset.name, dataset.path)

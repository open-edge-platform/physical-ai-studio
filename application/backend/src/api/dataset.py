from typing import Annotated

from schemas import LeRobotDatasetInfo, Episode
from fastapi import APIRouter, Depends
from services import DatasetService
from utils.dataset import get_local_repositories
from api.dependencies import get_dataset_service
from utils.dataset import get_dataset_episodes

router = APIRouter()


@router.get("/lerobot_datasets")
async def list_leorobot_datasets() -> list[LeRobotDatasetInfo]:
    """Get all local lerobot datasets from huggingface cache."""
    return [
        LeRobotDatasetInfo(
            root=str(dataset.root),
            repo_id=dataset.repo_id,
            total_episodes=dataset.total_episodes,
            total_frames=dataset.total_frames,
            fps=dataset.fps,
            features=list(dataset.features),
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

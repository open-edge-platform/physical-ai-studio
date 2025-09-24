from typing import Annotated

from schemas import Dataset, LeRobotDataset
from fastapi import APIRouter
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from utils.dataset import get_local_repositories

router = APIRouter()


@router.get("/lerobot_datasets")
async def list_leorobot_datasets() -> list[LeRobotDataset]:
    """Get all local lerobot datasets from huggingface cache"""
    return [LeRobotDataset(
        root=str(dataset.root),
        repo_id=dataset.repo_id,
        total_episodes=dataset.total_episodes,
        total_frames=dataset.total_frames,
        fps=dataset.fps,
        features=list(dataset.features)
        ) for dataset in get_local_repositories()
    ]

@router.get("")
async def list_datasets() -> list[Dataset]:
    pass

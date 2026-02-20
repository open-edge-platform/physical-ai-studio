from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from .base import BaseIDModel


class EpisodeInfo(BaseModel):
    episode_index: int
    tasks: list[str]
    length: int


class EpisodeVideo(BaseModel):
    start: float
    end: float
    path: str


class Episode(BaseModel):
    episode_index: int
    length: int
    fps: int
    tasks: list[str]
    actions: list[list[float]]
    action_keys: list[str]
    videos: dict[str, EpisodeVideo]
    thumbnail: str | None


class LeRobotDatasetInfo(BaseModel):
    root: str
    repo_id: str
    total_episodes: int
    total_frames: int
    features: list[str]
    fps: int
    robot_type: str


class Dataset(BaseIDModel):
    name: str = "Default Name"
    path: str
    project_id: Annotated[UUID, Field(description="Unique identifier")]
    environment_id: Annotated[UUID, Field(description="Unique identifier")]

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "fec4a691-76ee-4f66-8dea-aad3110e16d6",
                "name": "Collect blocks",
                "path": "/some/path/to/dataset",
                "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "environment_id": "e7d4bef8-158d-6c1g-c435-20ffb2chc675",
            }
        }
    }


class Snapshot(BaseIDModel):
    path: str
    dataset_id: Annotated[UUID, Field(description="Dataset Unique Identifier")]
    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "dataset_id": "fec4a691-76ee-4f66-8dea-aad3110e16d6",
                "path": "/some/path/to/snapshot",
            }
        }
    }

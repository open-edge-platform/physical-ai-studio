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


class Episode(BaseModel):
    episode_index: int
    length: int
    fps: int
    tasks: list[str]
    actions: list[list[float]]
    videos: dict[str, EpisodeVideo]
    modification_timestamp: int


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

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "fec4a691-76ee-4f66-8dea-aad3110e16d6",
                "name": "Collect blocks",
                "path": "/some/path/to/dataset",
                "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
            }
        }
    }

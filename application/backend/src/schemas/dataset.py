from pydantic import BaseModel
from schemas.base import BaseIDNameModel, BaseIDNameModel


class EpisodeInfo(BaseModel):
    episode_index: int
    tasks: list[str]
    length: int


class Episode(BaseModel):
    episode_index: int
    length: int
    fps: int
    tasks: list[str]
    actions: list[list[float]]
    modification_timestamp: int


class LeRobotDatasetInfo(BaseModel):
    root: str
    repo_id: str
    total_episodes: int
    total_frames: int
    features: list[str]
    fps: int


class Dataset(BaseIDNameModel):
    path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Collect blocks",
                "path": "/some/path/to/dataset",
            }
        }
    }

from pydantic import BaseModel, Field
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

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

class Dataset(BaseModel):
    repo_id: str
    episodes: list[Episode]
    total_frames: int
    features: list[str]
    fps: int

    @classmethod
    def fromLeRobotMetaData(cls, data: LeRobotDatasetMetadata) -> 'Dataset':
        """Convert LeRobotMetaData to Dataset"""
        return Dataset(
            repo_id=data.repo_id,
            episodes=data.total_episodes,
            total_frames=data.total_frames,
            features=data.features.keys(),
            fps=data.fps,
        )


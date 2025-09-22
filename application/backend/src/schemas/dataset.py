from pydantic import BaseModel


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
    tasks: list[str]
    fps: int

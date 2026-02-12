from git import rmtree
from uuid import uuid4
from pathlib import Path

from settings import get_settings
from internal_datasets.dataset_client import DatasetClient

class DeleteEpisodesMutation:
    cache_dir: Path
    source_dataset: DatasetClient
    def __init__(self, source_dataset: DatasetClient):
        settings = get_settings()
        self.cache_dir = settings.cache_dir / str(uuid4())
        self.source_dataset = source_dataset

    def delete_episodes(self, episode_indices: list[int]) -> DatasetClient:
        """Delete episodes."""
        self.cache_dataset = self.source_dataset.delete_episodes(episode_indices, self.cache_dir)
        self.source_dataset.overwrite(self.cache_dataset)

        self.teardown()
        return self.source_dataset

    def teardown(self) -> None:
        """Remove cache dir if it exists."""
        if self.cache_dir.is_dir():
            rmtree(self.cache_dir)

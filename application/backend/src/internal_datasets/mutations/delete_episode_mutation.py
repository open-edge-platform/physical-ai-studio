from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from internal_datasets.dataset_client import DatasetClient
from settings import get_settings


class DeleteEpisodesMutation:
    cache_dir: Path
    source_dataset: DatasetClient

    def __init__(self, source_dataset: DatasetClient):
        settings = get_settings()
        self.cache_dir = settings.cache_dir / str(uuid4())
        self.source_dataset = source_dataset

    def delete_episodes(self, episode_indices: list[int]) -> DatasetClient:
        """Delete episodes. If all episodes are removed it deletes the dataset.

        NOTE: LeRobot does not allow empty datasets. This is an implementation detail of DatasetClient.
        However the remaining flow of delete_episodes makes no sense when removing all episodes.
        Leaving until new implementation of dataset client arrives.
        """
        remaining_episodes = [
            episode for episode in self.source_dataset.get_episodes() if episode.episode_index not in episode_indices
        ]
        if remaining_episodes:
            self.cache_dataset = self.source_dataset.delete_episodes(episode_indices, self.cache_dir)
            self.source_dataset.overwrite(self.cache_dataset)
            self.teardown()
        else:
            self.source_dataset.delete()

        return self.source_dataset

    def teardown(self) -> None:
        """Remove cache dir if it exists."""
        if self.cache_dir.is_dir():
            rmtree(self.cache_dir)

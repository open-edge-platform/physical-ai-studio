from git import rmtree
from schemas import Episode
from shutil import copytree
from internal_datasets.lerobot.lerobot_dataset import InternalLeRobotDataset
from settings import get_settings
from uuid import uuid4
from pathlib import Path

class RecordingMutation:
    """ This is a mutation wrapper to handle dataset recording.

    if dataset exists:
        copy dataset to cache
    else create new dataset in cache
        add_episodes as you wish
    finalize or discard
    finalize -> remove original dataset and copy over new and remove cache
    discard -> remove cache
    """
    cache_dataset: InternalLeRobotDataset
    has_mutation: bool = False
    source_path: Path

    @classmethod
    def from_existing_dataset(cls, source_path: Path) -> "RecordingMutation":
        settings = get_settings()
        cache_dir = settings.cache_dir / str(uuid4())
        copytree(source_path, cache_dir)
        cache_dataset = InternalLeRobotDataset(cache_dir)
        return cls(source_path, cache_dataset)


    @classmethod
    def from_new_dataset(cls, source_path: Path, fps: int, features: dict, robot_type: str) -> "RecordingMutation":
        settings = get_settings()
        cache_dir = settings.cache_dir / str(uuid4())
        cache_dataset = InternalLeRobotDataset(cache_dir)
        cache_dataset.create(fps, features, robot_type)
        return cls(source_path, cache_dataset)

    def __init__(self, source_path: Path, cache_dataset: InternalLeRobotDataset):
        self.cache_dataset = cache_dataset
        self.source_path = source_path
        self.cache_dataset.prepare_for_writing()

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        self.cache_dataset.add_frame(obs, act, task)

    def save_episode(self, task: str) -> Episode:
        """Save current recording buffer as episode."""
        self.has_mutation = True
        return self.cache_dataset.save_episode(task)

    def discard_buffer(self) -> None:
        """Discard current recording buffer."""
        self.cache_dataset.discard_buffer()

    def teardown(self) -> None:
        """If mutation exists apply and then remove cache."""
        if self.has_mutation:
            self.cache_dataset.finalize()
            self._overwrite_source_dataset()
        self.cache_dataset.delete()

    def _overwrite_source_dataset(self):
        if self.source_path.is_dir():
            rmtree(self.source_path)
        copytree(self.cache_dataset.path, self.source_path)

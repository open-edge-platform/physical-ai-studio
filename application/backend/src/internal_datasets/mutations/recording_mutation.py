from internal_datasets.dataset_client import DatasetClient
from schemas import Episode


class RecordingMutation:
    """This is a mutation wrapper to handle dataset recording.

    if dataset exists:
        copy dataset to cache
    else create new dataset in cache
        add_episodes as you wish
    finalize or discard
    finalize -> remove original dataset and copy over new and remove cache
    discard -> remove cache
    """

    cache_dataset: DatasetClient
    source_dataset: DatasetClient
    has_mutation: bool = False

    def __init__(self, source_dataset: DatasetClient, cache_dataset: DatasetClient):
        self.cache_dataset = cache_dataset
        self.source_dataset = source_dataset
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
        print(f"Teardown recording mutation. Has mutation {self.has_mutation}")
        if self.has_mutation:
            self.cache_dataset.finalize()
            self.source_dataset.overwrite(self.cache_dataset)
        self.cache_dataset.delete()

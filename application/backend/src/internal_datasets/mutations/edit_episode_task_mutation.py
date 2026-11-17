from internal_datasets.dataset_client import DatasetClient, EpisodeNotFoundError
from schemas import Episode


class EditEpisodeTaskMutation:
    """Change the task an episode was recorded with, in place.

    Unlike deleting episodes this needs no copy of the dataset: only task metadata is rewritten,
    each file atomically, and the videos stay as they are.
    """

    source_dataset: DatasetClient

    def __init__(self, source_dataset: DatasetClient):
        self.source_dataset = source_dataset

    def edit_task(self, episode_index: int, task: str) -> Episode | None:
        """Make ``task`` the only task of the episode. Returns the updated episode, or ``None`` if it does not exist."""
        task = task.strip()
        if not task:
            raise ValueError("The task must not be empty")
        try:
            self.source_dataset.set_episode_task(episode_index, task)
        except EpisodeNotFoundError:
            return None
        return self.source_dataset.find_episode(episode_index)

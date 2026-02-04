from abc import ABC, abstractmethod

from schemas import Episode


class DatasetClient(ABC):
    type: str
    exists_on_disk: bool = False
    has_episodes: bool = False

    @abstractmethod
    def prepare_for_writing(self, number_of_threads: int) -> None:
        """Processes for writing episodes."""

    @abstractmethod
    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        """Add frame to recording buffer."""

    @abstractmethod
    def save_episode(self, task: str) -> Episode:
        """Save current recording buffer as episode."""

    @abstractmethod
    def discard_buffer(self) -> None:
        """Discard current recording buffer."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up dataset and delete if no episodes."""

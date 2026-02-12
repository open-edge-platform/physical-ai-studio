from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from schemas import Episode

if TYPE_CHECKING:
    from internal_datasets.mutations.recording_mutation import RecordingMutation


class DatasetClient(ABC):
    type: str
    exists_on_disk: bool = False
    has_episodes: bool = False

    @abstractmethod
    def prepare_for_writing(self) -> None:
        """Processes for writing episodes."""

    @abstractmethod
    def get_episodes(self) -> list[Episode]:
        """Get episodes of dataset."""

    @abstractmethod
    def get_tasks(self) -> list[str]:
        """Get Tasks in dataset."""

    @abstractmethod
    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get Video path of specific episode and camera."""

    @abstractmethod
    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create dataset."""

    @abstractmethod
    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> "RecordingMutation":
        """Start recording mutation."""

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

    @abstractmethod
    def delete(self) -> None:
        """Delete dataset."""

    @abstractmethod
    def finalize(self) -> None:
        """Finalize changes to dataset."""

    @abstractmethod
    def overwrite(self, source: "DatasetClient") -> None:
        """Overwrite dataset with given dataset."""

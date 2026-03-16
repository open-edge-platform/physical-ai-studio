from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from schemas import Episode, EpisodeInfo

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
    def get_episode_infos(self) -> list[EpisodeInfo]:
        """Get episode summaries without heavy frame/action data."""

    @abstractmethod
    def find_episode(self, episode_index: int) -> Episode | None:
        """Find episode by index or return None."""

    @abstractmethod
    def get_tasks(self) -> list[str]:
        """Get Tasks in dataset."""

    @abstractmethod
    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get Video path of specific episode and camera."""

    @abstractmethod
    def get_video_keys(self) -> list[str]:
        """Get the video keys used to record the dataset"""

    @abstractmethod
    def get_episode_thumbnail_png(
        self,
        episode_index: int,
        video_key: str,
        width: int = 320,
        height: int = 240,
    ) -> tuple[bytes, Path] | None:
        """Build a PNG thumbnail and return bytes with source video path."""

    @abstractmethod
    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create dataset."""

    @abstractmethod
    def delete_episodes(self, episode_indices: list[int], output_path: Path) -> "DatasetClient":
        """Copy over repo without given episode_indices to output_path."""

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

    @abstractmethod
    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> "RecordingMutation":
        """Start recording mutation."""

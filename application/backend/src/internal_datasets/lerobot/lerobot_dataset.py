import copy
import shutil
import time
from os import path, stat
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import make_default_processors
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STR
from loguru import logger

from internal_datasets.dataset_client import DatasetClient
from schemas import Episode, EpisodeVideo
from utils.dataset import robot_for_action_features


class InternalLeRobotDataset(DatasetClient):
    type: str = "lerobot"
    _path: Path
    _dataset: LeRobotDataset

    _teleop_action_processor: RobotProcessorPipeline
    _robot_action_processor: RobotProcessorPipeline
    _robot_observation_processor: RobotProcessorPipeline

    def __init__(self, path: Path):
        self._path = path
        if self._check_repository_exists(path):
            self._dataset = LeRobotDataset(str(uuid4()), path)
            self.exists_on_disk = True
            self.has_episodes = self._dataset.num_episodes > 0

        self._teleop_action_processor, self._robot_action_processor, self._robot_observation_processor = (
            make_default_processors()
        )

    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create LeRobot dataset."""
        if self._check_repository_exists(self._path):
            raise Exception(f"Dataset already exists at {self._path}")
        self._dataset = LeRobotDataset.create(
            repo_id=str(uuid4()), root=self._path, fps=fps, features=features, robot_type=robot_type, use_videos=True
        )
        self.exists_on_disk = True
        self.has_episodes = False

    def get_tasks(self) -> list[str]:
        """Get Tasks in dataset."""
        if not self.exists_on_disk:
            return []
        return list(self._dataset.meta.tasks.to_dict()["task_index"].keys())

    def get_video_path(self, episode: int, camera: str) -> Path:
        """Get Video path of specific episode and camera."""
        metadata = self._dataset.meta
        full_camera_name = f"observation.images.{camera}"
        return Path(metadata.root) / Path(metadata.get_video_file_path(episode, full_camera_name))

    def prepare_for_writing(self, number_of_threads: int) -> None:
        """Start image writer &"""
        self._dataset.start_image_writer(
            num_processes=0,
            num_threads=number_of_threads,
        )

    def get_episodes(self) -> list[Episode]:
        """Get episodes of dataset."""

        if not self.exists_on_disk:
            return []
        metadata = self._dataset.meta
        episodes = metadata.episodes

        result = []
        action_feature_names = self._dataset.features.get("action", {}).get("names", [])
        follower_robot = robot_for_action_features(action_feature_names)
        for episode in episodes:
            full_path = path.join(metadata.root, metadata.get_data_file_path(episode["episode_index"]))
            stat_result = stat(full_path)
            result.append(
                Episode(
                    actions=self._get_episode_actions(episode).tolist(),
                    fps=metadata.fps,
                    modification_timestamp=stat_result.st_mtime_ns // 1e6,
                    videos={
                        video_key: EpisodeVideo(
                            start=episode[f"videos/{video_key}/from_timestamp"],
                            end=episode[f"videos/{video_key}/to_timestamp"],
                        )
                        for video_key in self._dataset.meta.video_keys
                    },
                    follower_robot_types=[follower_robot],
                    action_keys=action_feature_names,
                    **episode,
                )
            )

        return result

    def add_frame(self, obs: dict, act: dict, task: str) -> None:
        """Add frame to recording buffer."""
        frame = self._process_frame(obs, act, task)
        self._dataset.add_frame(frame)

    def save_episode(self, task: str) -> Episode:
        """Save current recording buffer as episode."""
        new_episode = self._build_episode_from_buffer(self._dataset.meta.latest_episode, task)
        self._dataset.save_episode()
        return new_episode

    def discard_buffer(self) -> None:
        """Discard current recording buffer."""
        self._dataset.clear_episode_buffer()

    def teardown(self) -> None:
        """Finalize dataset or delete if no episodes."""
        # TODO: Implement a wait for when an episode is still being written, but teardown is called.
        if self._dataset.num_episodes == 0:
            logger.info("Removing dataset since it has no episodes")
            shutil.rmtree(self._path)
        else:
            logger.info("Finalizing")
            self._dataset.finalize()

    def _process_frame(self, obs: dict, act: dict, task: str) -> dict:
        obs_processed = self._robot_observation_processor(obs)
        act_processed_teleop = self._teleop_action_processor((act, obs))
        action_frame = build_dataset_frame(self._dataset.features, act_processed_teleop, prefix=ACTION)
        observation_frame = build_dataset_frame(self._dataset.features, obs_processed, prefix=OBS_STR)

        return {**observation_frame, **action_frame, "task": task}

    @staticmethod
    def _check_repository_exists(path: Path) -> bool:
        """Check if repository path contains info and therefor exists."""
        return (path / Path("meta/info.json")).is_file()

    def _build_episode_from_buffer(self, episode: dict | None, task: str) -> Episode:
        """Build Episode object from buffer and episode dict."""
        data = self._build_episode_data_from_buffer()
        if data is None or self._dataset is None:
            raise Exception("No dataset loaded.")

        end = data["timestamp"][-1]
        video_timestamps = {video_key: EpisodeVideo(start=0, end=end) for video_key in self._dataset.meta.video_keys}
        if episode is not None:
            for video_key in self._dataset.meta.video_keys:
                offset = episode[f"videos/{video_key}/to_timestamp"][-1]
                video_timestamps[video_key].start += offset
                video_timestamps[video_key].end += offset

        action_feature_names = self._dataset.features.get("action", {}).get("names", [])
        follower_robot = robot_for_action_features(action_feature_names)

        return Episode(
            episode_index=data["episode_index"].tolist()[0],
            length=len(data["frame_index"]),
            fps=self._dataset.fps,
            tasks=[task],
            actions=data["action"].tolist(),
            videos=video_timestamps,
            modification_timestamp=int(time.time()),
            action_keys=action_feature_names,
            follower_robot_types=[follower_robot],
        )

    def _build_episode_data_from_buffer(self) -> dict:
        """Build episode data from the buffer.

        LeRobotDataset V3 doesnt update episode data on save.
        In order to get the episode data we duplicate the actions that happen inside.
        """
        if self._dataset is None:
            raise Exception("No dataset loaded.")

        episode_buffer = copy.deepcopy(self._dataset.episode_buffer)
        if episode_buffer is None:
            raise Exception("Attempting to save episode, but no episode in buffer.")

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self._dataset.meta.total_frames, self._dataset.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self._dataset.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self._dataset.meta.get_task_index(task) for task in tasks])

        for key, ft in self._dataset.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        return episode_buffer

    def _get_episode_actions(self, episode: dict) -> torch.Tensor:
        """Get episode actions tensor from specific episode."""
        from_idx = episode["dataset_from_index"]
        to_idx = episode["dataset_to_index"]
        actions = self._dataset.hf_dataset["action"][from_idx:to_idx]
        return torch.stack(actions)

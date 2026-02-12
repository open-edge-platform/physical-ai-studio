from git import rmtree
from settings import get_settings
from internal_datasets.mutations.recording_mutation import RecordingMutation
import base64
import copy
import shutil
import cv2
from os import path
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
    path: Path
    _dataset: LeRobotDataset

    _teleop_action_processor: RobotProcessorPipeline
    _robot_action_processor: RobotProcessorPipeline
    _robot_observation_processor: RobotProcessorPipeline

    def __init__(self, dataset_path: Path):
        self.path = dataset_path
        if self._check_repository_exists(dataset_path):
            self._dataset = LeRobotDataset(str(uuid4()), dataset_path)
            self.has_episodes = self._dataset.num_episodes > 0

        self._teleop_action_processor, self._robot_action_processor, self._robot_observation_processor = (
            make_default_processors()
        )

    def create(self, fps: int, features: dict, robot_type: str) -> None:
        """Create LeRobot dataset."""
        if self._check_repository_exists(self.path):
            raise Exception(f"Dataset already exists at {self.path}")
        self._dataset = LeRobotDataset.create(
            repo_id=str(uuid4()), root=self.path, fps=fps, features=features, robot_type=robot_type, use_videos=True
        )
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

    def prepare_for_writing(self) -> None:
        """Start image writer &"""
        num_threads = 4 * len(self._dataset.meta.camera_keys)
        self._dataset.start_image_writer(
            num_processes=0,
            num_threads=num_threads,
        )

    def overwrite(self, source: DatasetClient) -> None:
        """Overwrite this dataset with the given dataset."""
        if not isinstance(source, InternalLeRobotDataset):
            raise ValueError(f"Cannot overwrite lerobot dataset with {source.__class__}")

        if self.path.is_dir():
            rmtree(self.path)

        shutil.copytree(source.path, self.path)

    def get_episodes(self) -> list[Episode]:
        """Get episodes of dataset."""

        if not self.exists_on_disk:
            return []
        metadata = self._dataset.meta
        episodes = metadata.episodes

        image_keys = self._dataset.meta.camera_keys
        image_key = image_keys[0]

        result = []
        action_feature_names = self._dataset.features.get("action", {}).get("names", [])
        follower_robot = robot_for_action_features(action_feature_names)
        for episode in episodes:
            episode_index = episode["episode_index"]
            thumbnail = self._build_thumbnail(episode, image_key) if len(image_keys) > 0 else None
            result.append(
                Episode(
                    actions=self._get_episode_actions(episode).tolist(),
                    fps=metadata.fps,
                    videos={
                        video_key: EpisodeVideo(
                            start=episode[f"videos/{video_key}/from_timestamp"],
                            end=episode[f"videos/{video_key}/to_timestamp"],
                            path=str(Path(metadata.root).stem / metadata.get_video_file_path(episode_index, video_key))
                        )
                        for video_key in self._dataset.meta.video_keys
                    },
                    follower_robot_types=[follower_robot],
                    action_keys=action_feature_names,
                    thumbnail=thumbnail,
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
            self.delete()
        else:
            logger.info("Finalizing")
            self.finalize()

    def delete(self) -> None:
        """Delete dataset."""
        shutil.rmtree(self.path)

    def finalize(self) -> None:
        """Finalize changes to dataset."""
        logger.info(f"Finalizing dataset {self.path}")
        self._dataset.finalize()

    def _process_frame(self, obs: dict, act: dict, task: str) -> dict:
        obs_processed = self._robot_observation_processor(obs)
        act_processed_teleop = self._teleop_action_processor((act, obs))
        action_frame = build_dataset_frame(self._dataset.features, act_processed_teleop, prefix=ACTION)
        observation_frame = build_dataset_frame(self._dataset.features, obs_processed, prefix=OBS_STR)

        return {**observation_frame, **action_frame, "task": task}

    def start_recording_mutation(self, fps: int, features: dict, robot_type: str) -> RecordingMutation:
        """Start recording mutation."""
        settings = get_settings()
        cache_dir = settings.cache_dir / str(uuid4())

        print(f"Creating cache dataset {cache_dir}")
        if self.exists_on_disk:
            shutil.copytree(self.path, cache_dir)
            cache_dataset = InternalLeRobotDataset(cache_dir)
        else:
            cache_dataset = InternalLeRobotDataset(cache_dir)
            cache_dataset.create(fps, features, robot_type)

        return RecordingMutation(self, cache_dataset)

    @property
    def exists_on_disk(self) -> bool:
        """Check if repo exists."""
        return self._check_repository_exists(self.path)

    @staticmethod
    def _check_repository_exists(repo_path: Path) -> bool:
        """Check if repository path contains info and therefor exists."""
        return (repo_path / "meta/info.json").is_file()

    def _build_episode_from_buffer(self, episode: dict | None, task: str) -> Episode:
        """Build Episode object from buffer and episode dict."""
        data = self._build_episode_data_from_buffer()
        if data is None or self._dataset is None:
            raise Exception("No dataset loaded.")

        end = data["timestamp"][-1]
        episode_index = data["episode_index"].tolist()[0]
        video_timestamps = {
            video_key: EpisodeVideo(start=0, end=end, path="") # TODO: Implement path
            for video_key in self._dataset.meta.video_keys
        }
        if episode is not None:
            for video_key in self._dataset.meta.video_keys:
                offset = episode[f"videos/{video_key}/to_timestamp"][-1]
                video_timestamps[video_key].start += offset
                video_timestamps[video_key].end += offset

        action_feature_names = self._dataset.features.get("action", {}).get("names", [])
        follower_robot = robot_for_action_features(action_feature_names)
        camera_key = self._dataset.meta.camera_keys[0]
        thumbnail = self._build_thumbnail_from_buffer(data, camera_key)
        return Episode(
            episode_index=episode_index,
            length=len(data["frame_index"]),
            fps=self._dataset.fps,
            tasks=[task],
            actions=data["action"].tolist(),
            videos=video_timestamps,
            action_keys=action_feature_names,
            follower_robot_types=[follower_robot],
            thumbnail=thumbnail
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

    def _build_thumbnail_from_buffer(self, episode_buffer: dict, image_key: str) -> str | None:
        thumbnail_size = (320,240)

        image_path = episode_buffer[image_key][-1]
        image = cv2.imread(image_path)
        if image is None:
            return None
        thumbnail = cv2.resize(image, thumbnail_size)
        _, imagebytes = cv2.imencode(".jpg", thumbnail)
        return base64.b64encode(imagebytes).decode()

    def _build_thumbnail(self, episode: dict, image_key: str) -> str:
        thumbnail_size = (320,240)

        from_idx = episode["dataset_from_index"]
        image = self._dataset[from_idx][image_key].permute(1, 2, 0).detach().numpy()
        rescaled = (image * 255).clip(0, 255).astype(np.uint8)
        resized = cv2.resize(rescaled, thumbnail_size)
        thumbnail = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        _, imagebytes = cv2.imencode(".jpg", thumbnail)
        return base64.b64encode(imagebytes).decode()

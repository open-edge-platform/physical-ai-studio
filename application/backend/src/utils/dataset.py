import os
import uuid
from os import listdir, path, stat
from pathlib import Path

import torch
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from schemas import CameraConfig, Dataset, Episode, EpisodeInfo, LeRobotDatasetInfo, ProjectConfig


def get_dataset_episodes(repo_id: str, root: str | None) -> list[Episode]:
    """Load dataset from LeRobot cache and get info"""
    dataset = LeRobotDataset(repo_id, root)
    metadata = dataset.meta
    episodes = metadata.episodes
    result = []
    for episode_index in episodes:
        full_path = path.join(metadata.root, metadata.get_data_file_path(episode_index))
        stat_result = stat(full_path)
        result.append(
            Episode(
                actions=get_episode_actions(dataset, episodes[episode_index]),
                fps=metadata.fps,
                modification_timestamp=stat_result.st_mtime_ns // 1e6,
                **episodes[episode_index],
            )
        )

    return result


def get_episode_actions(dataset: LeRobotDataset, episode: EpisodeInfo) -> torch.Tensor:
    """Get episode actions tensor from specific episode."""
    episode_index = episode["episode_index"]
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    actions = dataset.hf_dataset["action"][from_idx:to_idx]
    return torch.stack(actions)


def get_local_repository_ids(home: str | Path | None = None) -> list[str]:
    """Get all local repository ids."""
    home = Path(home) if home is not None else HF_LEROBOT_HOME

    repo_ids: list[str] = []
    for folder in listdir(home):
        if not os.path.isdir(home / folder):
            continue

        if folder == "calibration":
            continue

        owner = folder

        for repo in listdir(home / folder):
            if os.path.isdir(home / folder / repo):
                repo_ids.append(f"{owner}/{repo}")

    return repo_ids


def get_local_repositories(
    home: str | Path | None = None,
) -> list[LeRobotDatasetMetadata]:
    """Get all LeRobotDatasetMetaData for all local datasets."""
    home = Path(home) if home is not None else HF_LEROBOT_HOME

    result: list[LeRobotDatasetMetadata] = []
    for repo_id in get_local_repository_ids(home):
        try:
            metadata = LeRobotDatasetMetadata(repo_id, home / repo_id, None, force_cache_sync=False)
            result.append(metadata)
        except RepositoryNotFoundError:
            print(f"Could not find local repository online: {repo_id}")

    return result


def camera_config_from_dataset_features(
    dataset: LeRobotDatasetMetadata,
) -> list[CameraConfig]:
    """Build camera configs from existing LeRobotDatasetMetadata features."""
    return [
        CameraConfig(
            name=name.split(".")[-1],
            width=feature["info"]["video.width"],
            height=feature["info"]["video.height"],
            fps=feature["info"]["video.fps"],
            driver="webcam",
            use_depth=False,
            port_or_device_id="",
            id=uuid.uuid4()
        )
        for name, feature in dataset.features.items()
        if feature["dtype"] == "video"
    ]


def build_project_config_from_dataset(dataset: LeRobotDatasetInfo) -> ProjectConfig:
    """Build Project Config from LeRobotDatasetInfo."""
    metadata = LeRobotDatasetMetadata(dataset.repo_id, dataset.root)
    return ProjectConfig(
        fps=dataset.fps,
        cameras=camera_config_from_dataset_features(metadata),
        robot_type=dataset.robot_type,
        id=uuid.uuid4(),
    )


def build_dataset_from_lerobot_dataset(dataset: LeRobotDatasetInfo) -> Dataset:
    """Build dataset from LeRobotDatasetInfo."""
    return Dataset(name=dataset.repo_id, path=dataset.root, id=uuid.uuid4())

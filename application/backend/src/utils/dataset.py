import os
import uuid
from json.decoder import JSONDecodeError
from os import listdir, path, stat
from pathlib import Path

import torch
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME

from schemas import CameraConfig, Dataset, Episode, LeRobotDatasetInfo, ProjectConfig
from storage.storage import GETI_ACTION_DATASETS


def get_dataset_episodes(repo_id: str, root: str | None) -> list[Episode]:
    """Load dataset from LeRobot cache and get info"""
    if root and not check_repository_exists(Path(root)):
        return []
    dataset = LeRobotDataset(repo_id, root)
    metadata = dataset.meta
    episodes = metadata.episodes
    result = []
    for episode_index in episodes:
        full_path = path.join(metadata.root, metadata.get_data_file_path(episode_index))
        stat_result = stat(full_path)
        result.append(
            Episode(
                actions=get_episode_actions(dataset, episodes[episode_index]).tolist(),
                fps=metadata.fps,
                modification_timestamp=stat_result.st_mtime_ns // 1e6,
                **episodes[episode_index],
            )
        )

    return result


def get_episode_actions(dataset: LeRobotDataset, episode: dict) -> torch.Tensor:
    """Get episode actions tensor from specific episode."""
    episode_index = episode["episode_index"]
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    actions = dataset.hf_dataset["action"][from_idx:to_idx]
    return torch.stack(actions)


def list_directories(folder: Path) -> list[str]:
    """Get list of directories from folder."""
    res = []
    if os.path.isdir(folder):
        for candidate in listdir(folder):
            if os.path.isdir(folder / candidate):
                res.append(candidate)
    return res


def get_local_repository_ids(home: str | Path | None = None) -> list[str]:
    """Get all local repository ids."""
    home = Path(home) if home is not None else HF_LEROBOT_HOME

    repo_ids: list[str] = []
    for folder in list_directories(home):
        if folder == "calibration":
            continue

        owner = folder
        for repo in list_directories(home / folder):
            if os.path.isdir(home / folder / repo):
                repo_ids.append(f"{owner}/{repo}")

    return repo_ids


def get_geti_action_datasets(home: str | Path | None = None) -> list[tuple[str, Path]]:
    """Get all local repository ids."""
    home = Path(home) if home is not None else GETI_ACTION_DATASETS

    repo_ids: list[tuple[str, Path]] = []
    for repo in list_directories(home):
        if os.path.isdir(home / repo):
            repo_ids.append((repo, home / repo))

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
        except JSONDecodeError:
            print(f"Could not parse local repository: {repo_id}")

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
            id=uuid.uuid4(),
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


def build_dataset_from_lerobot_dataset(dataset: LeRobotDatasetInfo, project_id: uuid.UUID) -> Dataset:
    """Build dataset from LeRobotDatasetInfo."""
    return Dataset(name=dataset.repo_id, path=dataset.root, id=uuid.uuid4(), project_id=project_id)


def check_repository_exists(path: Path) -> bool:
    """Check if repository path contains info and therefor exists."""
    return (path / Path("meta/info.json")).is_file()

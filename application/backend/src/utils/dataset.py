import os
import traceback
import uuid
from json.decoder import JSONDecodeError
from os import listdir, path, stat
from pathlib import Path

import torch
from datasets.exceptions import DatasetGenerationError
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import HF_LEROBOT_HOME
from loguru import logger

from exceptions import ResourceInUseError, ResourceType
from schemas import Dataset, Episode, EpisodeVideo, LeRobotDatasetInfo
from settings import get_settings


def load_local_lerobot_dataset(path: str | None, **kwargs) -> LeRobotDataset:
    """Load local LeRobot Dataset

    Using unique repo id to prevent side effects from lerobot.
    """
    return LeRobotDataset(str(uuid.uuid4()), path, **kwargs)


def get_dataset_episodes(root: str | None) -> list[Episode]:
    """Load dataset from LeRobot cache and get info"""
    if root and not check_repository_exists(Path(root)):
        return []
    try:
        dataset = load_local_lerobot_dataset(root)
        metadata = dataset.meta
        episodes = metadata.episodes
        result = []
        for episode in episodes:
            full_path = path.join(metadata.root, metadata.get_data_file_path(episode["episode_index"]))
            stat_result = stat(full_path)
            result.append(
                Episode(
                    actions=get_episode_actions(dataset, episode).tolist(),
                    fps=metadata.fps,
                    modification_timestamp=stat_result.st_mtime_ns // 1e6,
                    videos={
                        video_key: build_episode_video_from_lerobot_episode_dict(episode, video_key)
                        for video_key in dataset.meta.video_keys
                    },
                    **episode,
                )
            )

        return result
    except DatasetGenerationError as e:
        raise ResourceInUseError(ResourceType.DATASET, str(e))
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        return []


def build_episode_video_from_lerobot_episode_dict(episode: dict, video_key: str) -> EpisodeVideo:
    """Build episode video data for specific episode."""
    return EpisodeVideo(
        start=episode[f"videos/{video_key}/from_timestamp"],
        end=episode[f"videos/{video_key}/to_timestamp"],
    )


def get_episode_actions(dataset: LeRobotDataset, episode: dict) -> torch.Tensor:
    """Get episode actions tensor from specific episode."""
    from_idx = episode["dataset_from_index"]
    to_idx = episode["dataset_to_index"]
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
    settings = get_settings()
    home = Path(home) if home is not None else settings.datasets_dir

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


def build_dataset_from_lerobot_dataset(dataset: LeRobotDatasetInfo, project_id: uuid.UUID) -> Dataset:
    """Build dataset from LeRobotDatasetInfo."""
    return Dataset(name=dataset.repo_id, path=dataset.root, id=uuid.uuid4(), project_id=project_id)


def check_repository_exists(path: Path) -> bool:
    """Check if repository path contains info and therefor exists."""
    return (path / Path("meta/info.json")).is_file()

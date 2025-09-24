from os import path, stat, listdir

from pathlib import Path
from typing import List
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.constants import HF_LEROBOT_HOME
from huggingface_hub.errors import RepositoryNotFoundError


from schemas import Dataset, Episode, EpisodeInfo


def get_dataset(repo_id: str) -> Dataset:
    """Load dataset from LeRobot cache and get info"""
    dataset = LeRobotDataset(repo_id)
    metadata = dataset.meta
    episodes = metadata.episodes
    result = Dataset(
        episodes=[],
        features=metadata.features.keys(),
        fps=metadata.fps,
        tasks=list(metadata.tasks.values()),
        repo_id=repo_id,
        total_frames=metadata.total_frames,
    )

    for episode_index in episodes:
        full_path = path.join(metadata.root, metadata.get_data_file_path(episode_index))
        stat_result = stat(full_path)
        result.episodes.append(
            Episode(
                actions=get_episode_actions(dataset, episodes[episode_index]),
                fps=metadata.fps,
                modification_timestamp=stat_result.st_mtime_ns // 1e6,
                **episodes[episode_index],
            )
        )

    return result


def get_episode_actions(dataset: LeRobotDataset, episode: EpisodeInfo) -> torch.Tensor:
    """Get episode actions tensor from specific episode"""
    episode_index = episode["episode_index"]
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    actions = dataset.hf_dataset["action"][from_idx:to_idx]
    return torch.stack(actions)


def get_local_repository_ids(home: str | Path | None = None) -> List[str]:
    home = Path(home) if home is not None else HF_LEROBOT_HOME

    repo_ids: List[str] = []
    for folder in listdir(home):
        if folder == "calibration":
            continue

        owner = folder

        for repo in listdir(home / folder):
            repo_ids.append(f"{owner}/{repo}")

    return repo_ids


def get_local_repositories(home: str | Path | None = None) -> List[LeRobotDatasetMetadata]:
    home = Path(home) if home is not None else HF_LEROBOT_HOME

    result: List[LeRobotDatasetMetadata] = []
    for repo_id in get_local_repository_ids(home):
        try:
            metadata = LeRobotDatasetMetadata(repo_id, home / repo_id, None, force_cache_sync=False)
            result.append(metadata)
        except RepositoryNotFoundError:
            print(f"Could not find local repository online: {repo_id}")

    return result

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from schemas import EpisodeInfo, Dataset, Episode
import torch
from os import path, stat

def get_dataset(repo_id: str) -> Dataset:
    """Load dataset from LeRobot cache and get info """
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
        result.episodes.append(Episode(
            actions = get_episode_actions(dataset, episodes[episode_index]),
            fps = metadata.fps,
            modification_timestamp = stat_result.st_mtime_ns // 1e6,
            **episodes[episode_index]
        ))

    return result


def get_episode_actions(dataset: LeRobotDataset, episode: EpisodeInfo) -> torch.Tensor:
    """Get episode actions tensor from specific episode"""
    episode_index = episode["episode_index"]
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()
    actions = dataset.hf_dataset["action"][from_idx:to_idx]
    actions_tensor = torch.stack(actions)
    return actions_tensor

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
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.utils.constants import HF_LEROBOT_HOME
from loguru import logger

from exceptions import ResourceInUseError, ResourceType
from robots.utils import get_robot_client
from schemas import CameraConfig, Dataset, Episode, EpisodeVideo, LeRobotDatasetInfo, ProjectConfig
from schemas.environment import EnvironmentWithRelations
from schemas.project_camera import Camera
from services.robot_calibration_service import RobotCalibrationService
from settings import get_settings
from utils.serial_robot_tools import RobotConnectionManager


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
    except RepositoryNotFoundError:
        return []
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
            fingerprint="",
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


async def get_camera_features(camera: Camera) -> tuple[int, int, int]:
    """Get features of a camera.

    Note: This works for 'now', but ip cameras etc should probably just get a frame before returning this.
    """
    return (camera.payload.height, camera.payload.width, 3)


async def build_observation_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
) -> dict:
    """Return dict of action features of environment."""
    if len(environment.robots) > 1:
        # TODO: Implement, should probably prefix feature the robots only when len(robots) > 1
        # One issue is that you need to know which is which, so probably need a name identifier for robots
        raise ValueError("Environments with multiple robots not implemented yet")

    output_features = await build_action_features(environment, robot_manager, calibration_service)
    for camera in environment.cameras:
        output_features[camera.name.lower()] = await get_camera_features(camera)

    return output_features


async def build_action_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
) -> dict:
    """Return dict of action features of environment."""
    output_features = {}
    for robot in environment.robots:
        client = await get_robot_client(robot.robot, robot_manager, calibration_service)
        for feature in client.features():
            output_features[feature] = float
    return output_features


async def build_lerobot_dataset_features(
    environment: EnvironmentWithRelations,
    robot_manager: RobotConnectionManager,
    calibration_service: RobotCalibrationService,
    use_videos: bool = True,
) -> dict:
    """Build lerobot dataset features."""
    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    observation_features = await build_observation_features(environment, robot_manager, calibration_service)
    action_features = await build_action_features(environment, robot_manager, calibration_service)

    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=use_videos,
        ),
    )

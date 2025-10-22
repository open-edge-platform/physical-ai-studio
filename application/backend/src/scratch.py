#!/usr/bin/env python3


from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from schemas import LeRobotDatasetInfo, Project
from utils.dataset import build_project_config_from_dataset, build_dataset_from_lerobot_dataset
from services import ProjectService

dataset = LeRobotDatasetMetadata("rhecker/duplo")
info = LeRobotDatasetInfo(
    root=str(dataset.root),
    repo_id=dataset.repo_id,
    total_episodes=dataset.total_episodes,
    total_frames=dataset.total_frames,
    fps=dataset.fps,
    features=list(dataset.features),
    robot_type=dataset.robot_type,
)

project_service = ProjectService()

update = {"config": build_project_config_from_dataset(info), "datasets": [build_dataset_from_lerobot_dataset(info)]}
project = project_service.create_project(Project(name="Test Project"))
project_service.update_project(project, update)

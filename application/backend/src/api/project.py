from typing import Annotated
from uuid import UUID
from pathlib import Path

from fastapi import APIRouter, Depends, status
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from api.dependencies import get_model_service, get_project_id, get_project_service
from exceptions import ResourceAlreadyExistsError
from schemas import LeRobotDatasetInfo, Model, Project, ProjectConfig, TeleoperationConfig, InferenceConfig
from services import ModelService, ProjectService
from utils.dataset import build_dataset_from_lerobot_dataset, build_project_config_from_dataset, check_repository_exists

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("")
async def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> list[Project]:
    """Fetch all projects."""
    return await project_service.get_project_list()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    project: Project,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Create a new project."""
    return await project_service.create_project(project)


@router.put("/{project_id}/project_config")
async def set_project_config(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_config: ProjectConfig,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Set project config."""
    project = await project_service.get_project_by_id(project_id)
    update = {
        "config": project_config,
    }
    return await project_service.update_project(project, update)


@router.post("/{project_id}/import_dataset", status_code=status.HTTP_201_CREATED)
async def import_dataset(
    project_id: Annotated[UUID, Depends(get_project_id)],
    lerobot_dataset: LeRobotDatasetInfo,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Set the project from a dataset, only available when config is None."""
    project = await project_service.get_project_by_id(project_id)
    update = {}
    if project.config is not None:
        raise ResourceAlreadyExistsError("project config", "Import disabled when project already has config.")
    if project.datasets:
        raise ResourceAlreadyExistsError("dataset", "Import disabled when project already has a dataset.")

    update["config"] = build_project_config_from_dataset(lerobot_dataset)
    update["datasets"] = [build_dataset_from_lerobot_dataset(lerobot_dataset, project_id)]
    return await project_service.update_project(project, update)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> None:
    """Delete a project."""
    await project_service.delete_project(project_id)


@router.get("/{project_id}")
async def get_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Get project by id."""
    return await project_service.get_project_by_id(project_id)


@router.get("/{project_id}/models")
async def get_project_models(
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> list[Model]:
    """Get all models of a project."""
    return await model_service.get_project_models(project_id)


@router.get("/example_teleoperation_config")
async def get_example_teleoperation_config() -> TeleoperationConfig:
    """Stub call to get definition in ui."""
    return TeleoperationConfig()

@router.get("/example_inference_config")
async def get_example_inference_config() -> InferenceConfig:
    """Stub call to get definition in ui."""
    return InferenceConfig()


@router.get("/{project_id}/tasks")
async def get_tasks_for_dataset(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> dict[str, list[str]]:
    """Get all dataset tasks of a project."""
    project = await project_service.get_project_by_id(project_id)
    return {
        dataset.name: list(LeRobotDatasetMetadata(dataset.name, dataset.path).tasks.values())
        for dataset in project.datasets
        if check_repository_exists(Path(dataset.path))
    }

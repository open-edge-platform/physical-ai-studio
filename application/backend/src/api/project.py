from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from api.dependencies import get_model_service, get_project_id, get_project_service
from schemas import InferenceConfig, Model, Project, TeleoperationConfig
from services import ModelService, ProjectService
from utils.dataset import check_repository_exists

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
        dataset.name: list(LeRobotDatasetMetadata(dataset.name, dataset.path).tasks.to_dict()["task_index"].keys())
        for dataset in project.datasets
        if check_repository_exists(Path(dataset.path))
    }

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from api.dependencies import get_project_id, get_project_service
from schemas import LeRobotDatasetInfo, Project, ProjectConfig, TeleoperationConfig
from services import ProjectService
from services.base import ResourceInUseError, ResourceNotFoundError
from utils.dataset import build_dataset_from_lerobot_dataset, build_project_config_from_dataset, check_repository_exists

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("")
async def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> list[Project]:
    """Fetch all projects."""
    return await project_service.get_project_list()


@router.post("")
async def create_project(
    project: Project,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Create a new project."""
    return project_service.create_project(project)


@router.post("/{project_id}/project_config")
async def set_project_config(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_config: ProjectConfig,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Set project config."""
    project = project_service.get_project_by_id(project_id)
    update = {
        "config": project_config,
    }
    return project_service.update_project(project, update)


@router.post("/{project_id}/import_dataset")
async def import_dataset(
    project_id: Annotated[UUID, Depends(get_project_id)],
    lerobot_dataset: LeRobotDatasetInfo,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Set the project from a dataset, only available when config is None."""
    project = project_service.get_project_by_id(project_id)
    update = {}
    if project.config is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Import disabled when project already has config."
        )
    if project.datasets:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Import disabled when project already has a dataset."
        )

    update["config"] = build_project_config_from_dataset(lerobot_dataset)
    update["datasets"] = [build_dataset_from_lerobot_dataset(lerobot_dataset, project_id)]
    return project_service.update_project(project, update)


@router.delete("/{project_id}")
async def delete_project(
    project_id: Annotated[UUID, Depends(get_project_id)],
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> None:
    """Delete a project."""
    try:
        project_service.delete_project_by_id(project_id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ResourceInUseError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/{id}")
async def get_project(id: str, project_service: Annotated[ProjectService, Depends(get_project_service)]) -> Project:
    """Get project by id."""
    try:
        return await project_service.get_project_by_id(id)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/example_teleoperation_config")
async def get_example_teleoperation_config() -> TeleoperationConfig:
    """Stub call to get definition in ui, probably will be used later."""
    return TeleoperationConfig()


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
        if check_repository_exists(dataset.path)
    }

from typing import Annotated
from uuid import UUID

from api.dependencies import get_project_id, get_project_service
from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from schemas import LeRobotDatasetInfo, Project
from services import ProjectService
from services.base import ResourceInUseError, ResourceNotFoundError
from services.mappers.datasets_mapper import DatasetMapper
from services.mappers.project_mapper import ProjectConfigMapper
from utils.dataset import build_project_config_from_dataset

router = APIRouter()


@router.get("")
async def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> list[Project]:
    """Fetch all projects."""
    projects = project_service.list_projects()
    return projects


@router.put("")
async def create_project(
    project: Project,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Create a new project."""
    return project_service.create_project(project)


@router.put("/{project_id}/import_dataset")
async def import_dataset(
    project_id: Annotated[UUID, Depends(get_project_id)],
    lerobot_dataset: LeRobotDatasetInfo,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
) -> Project:
    """Set the project from a dataset, only available when config is None."""
    project = project_service.get_project_by_id(project_id)
    update = {}
    if project.config is None:
        update["config"] = build_project_config_from_dataset(lerobot_dataset)
    update["datasets"] = [DatasetMapper.from_lerobot_dataset(lerobot_dataset)]
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
async def get_project(
    id: str, project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> Project:
    """Get project by id."""
    return project_service.get_project_by_id(id)

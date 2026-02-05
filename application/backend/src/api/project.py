from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import get_model_service, get_project_id, get_project_service
from internal_datasets.utils import get_internal_dataset
from schemas import InferenceConfig, Model, Project, ProjectConfig, TeleoperationConfig
from services import ModelService, ProjectService

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
    res = {}

    for dataset in project.datasets:
        res[dataset.name] = get_internal_dataset(dataset).get_tasks()

    return res

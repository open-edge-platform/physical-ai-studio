from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import get_environment_id, get_environment_service, get_project_id
from schemas.environment import Environment, EnvironmentWithRelations
from services.environment_service import EnvironmentService

router = APIRouter(prefix="/api/projects/{project_id}/environments", tags=["Project Environments"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


@router.get("")
async def list_project_environments(
    project_id: ProjectID,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> list[Environment]:
    """Fetch all environments."""
    return await environment_service.get_environment_list(project_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project_environment(
    project_id: ProjectID,
    environment: Environment,
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> Environment:
    """Create a new environment."""
    return await environment_service.create_environment(project_id, environment)


@router.get("/{environment_id}")
async def get_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> EnvironmentWithRelations | None:
    """Get environment by id with eager loaded robots and cameras."""
    return await environment_service.get_environment_by_id(project_id, environment_id)


@router.put("/{environment_id}")
async def update_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    environment: Environment,
) -> EnvironmentWithRelations:
    """Update environment."""
    environment_with_id = environment.model_copy(update={"id": environment_id})

    return await environment_service.update_environment(
        project_id,
        environment_with_id,
    )


@router.delete("/{environment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_environment(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
) -> None:
    """Delete an environment."""
    await environment_service.delete_environment(project_id, environment_id)

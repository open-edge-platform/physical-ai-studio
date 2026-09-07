from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
from physicalai.config import to_yaml

from api.dependencies import RobotClientFactoryDep, get_environment_id, get_environment_service, get_project_id
from runtime.config_builder import RUNTIME_FPS, build_runtime_config, runtime_config_change_me
from schemas.environment import Environment, EnvironmentWithRelations, TeleoperatorRobotWithRobot
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
) -> EnvironmentWithRelations:
    """Get environment by id with eager loaded robots and cameras."""
    return await environment_service.get_environment_by_id(project_id, environment_id)


@router.get("/{environment_id}/runtime-config")
async def get_runtime_config(
    project_id: Annotated[UUID, Depends(get_project_id)],
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    environment_service: Annotated[EnvironmentService, Depends(get_environment_service)],
    robot_client_factory: RobotClientFactoryDep,
) -> Response:
    """Download a runnable physicalai teleoperation configuration."""
    environment = await environment_service.get_environment_by_id(project_id, environment_id)
    if len(environment.robots) != 1:
        raise HTTPException(status_code=400, detail="Runtime export requires exactly one follower robot")
    relation = environment.robots[0]
    if not isinstance(relation.tele_operator, TeleoperatorRobotWithRobot) or relation.tele_operator.robot is None:
        raise HTTPException(status_code=400, detail="Runtime teleoperation export requires a leader robot")

    try:
        document = await build_runtime_config(
            follower=relation.robot,
            leader=relation.tele_operator.robot,
            cameras=environment.cameras,
            robot_factory=robot_client_factory,
            fps=RUNTIME_FPS,
            # The rig this describes does not have to be attached to export it.
            allow_stored_port=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    unresolved = runtime_config_change_me(document)
    comments = "".join(f"# CHANGE_ME: replace machine-specific device path {path}\n" for path in unresolved)
    return Response(
        content=comments + to_yaml(document),
        media_type="application/yaml",
        headers={"Content-Disposition": 'attachment; filename="runtime.yaml"'},
    )


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

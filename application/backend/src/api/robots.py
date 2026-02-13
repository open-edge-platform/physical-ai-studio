from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import get_project_id, get_robot_id, get_robot_service
from schemas import Robot
from schemas.robot import RobotWithConnectionState
from services import RobotService

router = APIRouter(prefix="/api/projects/{project_id}/robots", tags=["Project Robots"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


@router.get("")
async def list_project_robots(
    project_id: ProjectID,
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
) -> list[Robot]:
    """Fetch all robots."""
    return await robot_service.get_robot_list(project_id)


@router.get("/online")
async def list_online_project_robots(
    project_id: ProjectID,
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
) -> list[RobotWithConnectionState]:
    """Fetch all robots."""
    return await robot_service.find_online_robots(project_id)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project_robot(
    project_id: ProjectID,
    robot: Robot,
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
) -> Robot:
    """Create a new robot."""
    return await robot_service.create_robot(project_id, robot)


@router.get("/{robot_id}")
async def get_project_robot(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
) -> Robot:
    """Get robot by id."""
    return await robot_service.get_robot_by_id(project_id, robot_id)


@router.put("/{robot_id}")
async def update_project_robot(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
    robot: Robot,
) -> Robot:
    """Set robot."""
    robot_with_id = robot.model_copy(update={"id": robot_id})

    return await robot_service.update_robot(
        project_id,
        robot_with_id,
    )


@router.delete("/{robot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_robot(
    project_id: Annotated[UUID, Depends(get_project_id)],
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    robot_service: Annotated[RobotService, Depends(get_robot_service)],
) -> None:
    """Delete a robot."""
    await robot_service.delete_robot(project_id, robot_id)

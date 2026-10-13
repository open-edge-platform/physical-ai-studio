# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A simulated robot's simulation: what it serves, and adding its cameras to an environment."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from api.dependencies import (
    AsyncSessionDep,
    RobotCatalogServiceDep,
    SimulationManagerDep,
    get_environment_id,
    get_project_id,
    get_robot_id,
)
from schemas.project_camera import IPCamera
from schemas.simulation import AddSimulationCamerasRequest, ProjectRobotSimulation, RobotSimulation
from services.robot_simulation_service import RobotSimulationService

router = APIRouter(prefix="/api/projects/{project_id}", tags=["Simulation"])

ProjectID = Annotated[UUID, Depends(get_project_id)]


def get_robot_simulation_service(
    session: AsyncSessionDep, catalog_service: RobotCatalogServiceDep, manager: SimulationManagerDep
) -> RobotSimulationService:
    """Provide a RobotSimulationService for the request's database session."""
    return RobotSimulationService(session, catalog_service.registry, manager)


RobotSimulationServiceDep = Annotated[RobotSimulationService, Depends(get_robot_simulation_service)]


@router.get("/robots/{robot_id}/simulation")
async def get_robot_simulation(
    project_id: ProjectID,
    robot_id: Annotated[UUID, Depends(get_robot_id)],
    service: RobotSimulationServiceDep,
) -> RobotSimulation:
    """Whether the robot's simulation runs, who started it, and its viewer, speed and cameras."""
    return await service.get(project_id, robot_id)


@router.get("/simulations")
async def list_project_simulations(
    project_id: ProjectID, service: RobotSimulationServiceDep
) -> list[ProjectRobotSimulation]:
    """The running simulations of the project's simulated robots, started by Studio or elsewhere."""
    return await service.list_running(project_id)


@router.post("/environments/{environment_id}/simulation-cameras", status_code=status.HTTP_201_CREATED)
async def add_simulation_cameras(
    project_id: ProjectID,
    environment_id: Annotated[UUID, Depends(get_environment_id)],
    request: AddSimulationCamerasRequest,
    service: RobotSimulationServiceDep,
) -> list[IPCamera]:
    """Add cameras of a running simulation to an environment, as IP cameras on its streams."""
    return await service.add_cameras(project_id, environment_id, request.robot_id, request.cameras, request.names)

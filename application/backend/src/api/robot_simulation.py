# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A simulated robot's simulation: what it serves, and adding its cameras to an environment."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from api.dependencies import AsyncSessionDep, RobotCatalogServiceDep, SimulationManagerDep
from schemas.project_camera import IPCamera
from schemas.simulation import AddSimulationCamerasRequest, ProjectRobotSimulation, RobotSimulation, SimulationCommand
from services.robot_simulation_service import RobotSimulationService

router = APIRouter(prefix="/api/projects/{project_id}", tags=["Simulation"])

ProjectID = UUID


def get_robot_simulation_service(
    session: AsyncSessionDep, catalog_service: RobotCatalogServiceDep, manager: SimulationManagerDep
) -> RobotSimulationService:
    """Provide a RobotSimulationService for the request's database session."""
    return RobotSimulationService(session, catalog_service.registry, manager)


RobotSimulationServiceDep = Annotated[RobotSimulationService, Depends(get_robot_simulation_service)]


@router.get("/robots/{robot_id}/simulation")
async def get_robot_simulation(
    project_id: ProjectID,
    robot_id: UUID,
    service: RobotSimulationServiceDep,
) -> RobotSimulation:
    """Whether the robot's simulation runs, who started it, and its viewer, speed and cameras."""
    return await service.get(project_id, robot_id)


@router.get("/robots/{robot_id}/simulation/command")
async def get_robot_simulation_command(
    project_id: ProjectID,
    robot_id: UUID,
    service: RobotSimulationServiceDep,
    scene: Annotated[str, Query(description="Id of one of the robot type's simulation scenes")],
    seed: Annotated[int | None, Query(ge=0, description="Fixed seed for the scene; none for a random layout")] = None,
) -> SimulationCommand:
    """The command that starts the robot's simulation in a terminal, outside Studio.

    Studio attaches to that simulation as one started outside Studio; it keeps running when Studio
    stops. 404 when the robot type's plugin offers no such command.
    """
    return await service.command(project_id, robot_id, scene, seed)


@router.get("/simulations")
async def list_project_simulations(
    project_id: ProjectID, service: RobotSimulationServiceDep
) -> list[ProjectRobotSimulation]:
    """The running simulations of the project's simulated robots, started by Studio or elsewhere."""
    return await service.list_running(project_id)


@router.post("/environments/{environment_id}/simulation-cameras", status_code=status.HTTP_201_CREATED)
async def add_simulation_cameras(
    project_id: ProjectID,
    environment_id: UUID,
    request: AddSimulationCamerasRequest,
    service: RobotSimulationServiceDep,
) -> list[IPCamera]:
    """Add cameras of a running simulation to an environment, as IP cameras on its streams."""
    return await service.add_cameras(project_id, environment_id, request.robot_id, request.cameras, request.names)
